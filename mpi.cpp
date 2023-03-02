#include "common.h"
#include <mpi.h>
#include <cmath>
#include <iostream>
#include <vector>
#include <list>

// Apply the force from neighbor to particle
void apply_force(particle_t& particle, particle_t& neighbor) {
    // Calculate Distance
    double dx = neighbor.x - particle.x;
    double dy = neighbor.y - particle.y;
    double r2 = dx * dx + dy * dy;

    // Check if the two particles should interact
    if (r2 > cutoff * cutoff)
        return;
    r2 = fmax(r2, min_r * min_r);
    double r = sqrt(r2);

    // Very simple short-range repulsive force
    double coef = (1 - cutoff / r) / r2 / mass;
    particle.ax += coef * dx;
    particle.ay += coef * dy;
}

// Integrate the ODE
void move(particle_t& p, double size) {
    // Slightly simplified Velocity Verlet integration
    // Conserves energy better than explicit Euler method
    p.vx += p.ax * dt;
    p.vy += p.ay * dt;
    p.x += p.vx * dt;
    p.y += p.vy * dt;

    // Bounce from walls
    while (p.x < 0 || p.x > size) {
        p.x = p.x < 0 ? -p.x : 2 * size - p.x;
        p.vx = -p.vx;
    }

    while (p.y < 0 || p.y > size) {
        p.y = p.y < 0 ? -p.y : 2 * size - p.y;
        p.vy = -p.vy;
    }
}

// Put any static global variables here that you will use throughout the simulation.

static int nbinsx;  // number of bins in one dimension (total number of bins = nbinsx^2)
static double dxbin; // length&width of each bin

static int mybinx_min,mybinx_max; // index of min (included) and max (excluded) bin row
static int mybinx_min_g,mybinx_max_g; // index of min (included) and max (excluded) bin row, taking into account ghost bins
static int binxthresh_up,binxthresh_dn; // index of max bin row held by the processor above the current one, and min bin row held by the processor below

static std::vector<std::vector<std::list<int>>> bins; //bins pertaining to one processor + ghost bins
static std::vector<std::vector<std::list<int>::iterator>> heads;

static int max_sent_parts;
static particle_t* send_up; //array of particles to be transferred up
static particle_t* send_dn; //array of particles to be transferred down
static int nparts_up;
static int nparts_dn;

void init_simulation(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
	// You can use this space to initialize data objects that you may need
	// This function will be called once before the algorithm begins
	// Do not do any particle simulation here    
    nbinsx = ((int)((double) size / (cutoff)) + 1);  // bin size greater or equal to cutoff length
    dxbin = size / (double) nbinsx;
    
    
    int nbins_per_proc_x[num_procs]; // number of rows of bins held by each processor
    int binxmin[num_procs+1],binxmax[num_procs]; // indices of min/max rows held by each processor
    binxmin[0] = 0;
    for (int p=0; p<num_procs;++p){
        nbins_per_proc_x[p] = (nbinsx + p) / num_procs; // arithmetic to make sure all bins are taken
        binxmin[p+1] = binxmin[p] + nbins_per_proc_x[p];
        binxmax[p] = fmin(nbinsx,binxmin[p+1]);
    }
    
    mybinx_min = binxmin[rank];
    mybinx_max = binxmax[rank];
    // add ghost bins
    mybinx_min_g = fmax(0,mybinx_min-1);
    mybinx_max_g = fmin(nbinsx,mybinx_max+1);
    // compute indices of rows held by neighboring processors
    binxthresh_up = (mybinx_min_g==0) ? -1 : mybinx_min-1;
    binxthresh_dn = (mybinx_max_g==nbinsx) ? nbinsx+1 : mybinx_max;
    
    /// Allocate memory for bins in this processor only -- plus ghost bins ///
    for (int ib = mybinx_min_g; ib < mybinx_max_g; ++ib) {
        std::vector<std::list<int>> row;
        bins.push_back(row);
        std::vector<std::list<int>::iterator> row2;
        heads.push_back(row2);
        for (int jb = 0; jb < nbinsx; ++jb) {
            std::list<int> list = {};
            bins[ib-mybinx_min_g].push_back(list);
            std::list<int>::iterator hd = list.begin();
            heads[ib-mybinx_min_g].push_back(hd);
        }
    }
    
    max_sent_parts = 3*num_parts/num_procs;
    send_up = new particle_t[max_sent_parts] ;
    send_dn = new particle_t[max_sent_parts] ;
    nparts_up = 0;
    nparts_dn = 0;
    
//    if(rank==0) std::cout << "nbinsx:" << nbinsx <<  std::endl;
//    std::cout << "rank:" << rank << " " << mybinx_min << " - "  << mybinx_max - 1 << std::endl;
    
  
    int count=0;
    /// INITIALIZE - compute which bin each particle is in, and add it if the bin pertains to the current processor  ///
    for (int i = 0; i < num_parts; ++i) {
        int ib = (int)(parts[i].x / dxbin);
        int jb = (int)(parts[i].y / dxbin);
        if ((ib >= mybinx_min_g) && (ib < mybinx_max_g)){
            bins[ib-mybinx_min_g][jb].emplace_front(i);
        }
    }
}


int check_num_parts(int rank){
    int count=0;
    for (int ib = mybinx_min; ib < mybinx_max; ++ib) {
        for (int jb = 0; jb < nbinsx; ++jb) {
            for (int i : bins[ib-mybinx_min_g][jb]) {
                count++;
            }
        }
    }
    std::cout << "rank :" << rank << " | count:" << count << std::endl;
    return count;
}


//int switchh=0;
static int count = 0;


void simulate_one_step(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
//    if(switchh==1){return;}
//    int count = check_num_parts(rank);
//    if((rank==0) && (count != 535)){std::cout << "shit" << std::endl;switchh=1;}
//    MPI_Barrier(MPI_COMM_WORLD);
//    if(rank==0) {count++;std::cout << "begin " << count << std::endl;}
    ////////////////////////////////////////////////////
    /////// Compute Forces -- exclude ghost bins ///////
    ////////////////////////////////////////////////////
    for (int ib = mybinx_min; ib < mybinx_max; ++ib) {
        for (int jb = 0; jb < nbinsx; ++jb) {
            // Keep track of list heads for the next step
            heads[ib-mybinx_min_g][jb] = bins[ib-mybinx_min_g][jb].begin();
            for (int i : bins[ib-mybinx_min_g][jb]) {
                parts[i].ax = parts[i].ay = 0;
                for (int local_i = fmax(0,ib-1); local_i <= fmin(nbinsx-1,ib+1); ++local_i){
                    for (int local_j = fmax(0,jb-1); local_j <= fmin(nbinsx-1,jb+1); ++local_j){
                        for (int j : bins[local_i-mybinx_min_g][local_j]) {
                                apply_force(parts[i], parts[j]);
                        }
                    }
                }
            }
        }
    }
    
//    std::cout << "rank:" << rank << std::endl;
//    MPI_Barrier(MPI_COMM_WORLD);
    
    ////////////////////////////////////////////////////////////////////////////////////////////////////
    ////// Move Particles & keep track of particles that move out of the scope of this processor ///////
    ////////////////////////////////////////////////////////////////////////////////////////////////////
    
    
    // Move Particles
    for (int ib = mybinx_min; ib < mybinx_max; ++ib) {
        for (int jb = 0; jb < nbinsx; ++jb) {
            std::list<int>::iterator it = heads[ib-mybinx_min_g][jb];
            while (it != bins[ib-mybinx_min_g][jb].end()){
                int i = *it;
                move(parts[i], size);
                int new_ib = (int)(parts[i].x / dxbin);
                int new_jb = (int)(parts[i].y / dxbin);
                
                if (ib != new_ib || jb != new_jb) {
                    it = bins[ib-mybinx_min_g][jb].erase( it );
                    if ((new_ib >= mybinx_min) && (new_ib < mybinx_max)){
                        bins[new_ib-mybinx_min_g][new_jb].emplace_front(i);
                    }
                    if (new_ib <= binxthresh_up) { // add to array to be communicated to processor above
//                        std::cout << "rank" << rank << " sending part " << i << " up" << std::endl;
                        send_up[nparts_up] = parts[i];
                        nparts_up++;
                    }
                    else if (new_ib >= binxthresh_dn) { // add to array to be communicated to processor above
//                        std::cout << "rank" << rank << " sending part " << i << " down" << std::endl;
                        send_dn[nparts_dn] = parts[i];
                        nparts_dn++;
                    }
                }
                else{
                    it ++;
                }
            }
        }
    }
    
//    std::cout << "rank:" << rank << " | nparts_up:" << nparts_up << " | nparts_dn:" << nparts_dn << " | max_sent_parts:" << max_sent_parts << std::endl;
//    MPI_Barrier(MPI_COMM_WORLD);

    
    ///////////////////////////////////////////
    // Communicate particles that have moved
    ///////////////////////////////////////////
    // send down & receive from above, then send up and receive from below
    particle_t recv_above[max_sent_parts];
    int nparts_above = 0;
    particle_t recv_below[max_sent_parts];
    int nparts_below = 0;
    MPI_Status status;
    
    if (mybinx_max_g<nbinsx){
        MPI_Send(send_dn, nparts_dn, PARTICLE, rank+1, 0, MPI_COMM_WORLD);
    }

    if (mybinx_min_g > 0){
        MPI_Recv(recv_above, max_sent_parts, PARTICLE, rank-1, 0, MPI_COMM_WORLD,&status);     
        MPI_Get_count(&status, PARTICLE, &nparts_above);
        
        MPI_Send(send_up, nparts_up, PARTICLE, rank-1, 1, MPI_COMM_WORLD);
    }

    if (mybinx_max_g<nbinsx){        
        MPI_Recv(recv_below, max_sent_parts, PARTICLE, rank+1, 1, MPI_COMM_WORLD,&status);
        MPI_Get_count(&status, PARTICLE, &nparts_below);
    }
//    std::cout << "rank:" << rank << " | nparts_above:" << nparts_above << " | nparts_below:" << nparts_below  << std::endl;
//    MPI_Barrier(MPI_COMM_WORLD);
    

    // Emplace received particles
    for (int i = 0; i < nparts_above; ++i){
        int parti = recv_above[i].id-1; // For some reason the ids are defined as i+1 in main.cpp
        parts[parti] = recv_above[i];
        int ib = (int)(recv_above[i].x / dxbin);
        int jb = (int)(recv_above[i].y / dxbin);
        if ((ib >= mybinx_min) && (ib < mybinx_max)){ //this check shouldn't be needed
//            if(i>1000){std::cout << "SHIIIIIIT1"  << std::endl;}
            bins[ib-mybinx_min_g][jb].emplace_front(parti); 
        }
    }

    for (int i = 0; i < nparts_below; ++i){
        int parti = recv_below[i].id-1; // For some reason the ids are defined as i+1 in main.cpp
        parts[parti] = recv_below[i];
        int ib = (int)(recv_below[i].x / dxbin);
        int jb = (int)(recv_below[i].y / dxbin);
        if ((ib >= mybinx_min) && (ib < mybinx_max)){ //this check shouldn't be needed
//            if(i>1000){std::cout << "SHIIIIII2"  << std::endl;}
            bins[ib-mybinx_min_g][jb].emplace_front(parti); 
        }        
    }
    
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /// Communicate ghost bins -- again, send down & receive from above, then send up and receive from below ///
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////
    nparts_up = 0;
    nparts_dn = 0; 
    nparts_above = 0;
    nparts_below = 0;
    
    if (mybinx_max_g<nbinsx){
        for (int jb = 0; jb < nbinsx; ++jb) {
            for (int i : bins[mybinx_max-1-mybinx_min_g][jb]) {
                send_dn[nparts_dn] = parts[i];
                nparts_dn++;
            }
            // empty my ghost bins
            bins[mybinx_max_g-1-mybinx_min_g][jb].clear();
        }
    }
    if (mybinx_min_g > 0){
        for (int jb = 0; jb < nbinsx; ++jb) {
            for (int i : bins[1][jb]) {
                send_up[nparts_up] = parts[i];
                nparts_up++;
            }
            // empty my ghost bins
            bins[0][jb].clear();
        }
    }
//    std::cout << "rank:" << rank << " | npartsghost_dn:" << nparts_dn << " | npartsghost_up:" << nparts_up  << std::endl;
//    MPI_Barrier(MPI_COMM_WORLD);
    
    if (mybinx_max_g<nbinsx){
        MPI_Send(send_dn, nparts_dn, PARTICLE, rank+1, 0, MPI_COMM_WORLD);
    }
    if (mybinx_min_g > 0){
        MPI_Recv(recv_above, max_sent_parts, PARTICLE, rank-1, 0, MPI_COMM_WORLD,&status);     
        MPI_Get_count(&status, PARTICLE, &nparts_above);
        
        MPI_Send(send_up, nparts_up, PARTICLE, rank-1, 1, MPI_COMM_WORLD);
    }
    if (mybinx_max_g<nbinsx){        
        MPI_Recv(recv_below, max_sent_parts, PARTICLE, rank+1, 1, MPI_COMM_WORLD,&status);     
        MPI_Get_count(&status, PARTICLE, &nparts_below);
    }
//    std::cout << "rank:" << rank << " | npartsghost_above:" << nparts_above << " | npartsghost_below:" << nparts_below  << std::endl;
//    MPI_Barrier(MPI_COMM_WORLD);

    // Emplace received particles in ghost bins
    for (int i = 0; i < nparts_above; ++i){
        int parti = recv_above[i].id-1; // For some reason the ids are defined as i+1 in main.cpp
        parts[parti] = recv_above[i];
        int ib = (int)(recv_above[i].x / dxbin);
        int jb = (int)(recv_above[i].y / dxbin);
//        if ((ib >= mybinx_min_g) && (ib < mybinx_max_g)){ //this check shouldn't be needed
        if (ib == mybinx_min_g){
            bins[0][jb].emplace_front(parti); 
        }
        else{
            std::cout << "shit " << ib -mybinx_min_g << std::endl;
        }
    }
    
    for (int i = 0; i < nparts_below; ++i){
        int parti = recv_below[i].id-1; // For some reason the ids are defined as i+1 in main.cpp
        parts[parti] = recv_below[i];
        int ib = (int)(recv_below[i].x / dxbin);
        int jb = (int)(recv_below[i].y / dxbin);
//        if ((ib >= mybinx_min_g) && (ib < mybinx_max_g)){ //this check shouldn't be needed
        if ((ib == mybinx_max_g-1)){
            bins[ib-mybinx_min_g][jb].emplace_front(parti); 
        }      
        else{
            std::cout << "shit " << ib-mybinx_max_g-1 << std::endl;
        }
    }   

    // Prepare for next step - no need to actually empty the buffers that get sent, just reset their counters to 0
    nparts_up = 0;
    nparts_dn = 0;    
//    MPI_Barrier(MPI_COMM_WORLD);
//    if(rank==0) {std::cout << "end " << count << std::endl;}
}




void gather_for_save(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    // Write this function such that at the end of it, the master (rank == 0)
    // processor has an in-order view of all particles. That is, the array
    // parts is complete and sorted by particle id.
    nparts_up=0;
    for (int ib = mybinx_min; ib < mybinx_max; ++ib) {
        for (int jb = 0; jb < nbinsx; ++jb) {
            for (int i : bins[ib-mybinx_min_g][jb]) {
                send_up[nparts_up] = parts[i];
                nparts_up++;
            }
        }
    }
    
    int recvcounts[num_procs];
    MPI_Gather(&nparts_up, 1, MPI_INT, recvcounts, 1, MPI_INT, 0,MPI_COMM_WORLD);
    
//    if (rank==0){std::cout << "rank:" << rank << " recv1 " << recvcounts[0] << " recv2 " << recvcounts[1] << std::endl;}
    int displacements[num_procs];
    displacements[0]=0;
    for (int p=1; p<num_procs;++p){
        displacements[p] = displacements[p-1] + recvcounts[p-1];
    }
    int tot_size = displacements[num_procs-1] + recvcounts[num_procs-1]; // should be == num_parts
    
    particle_t recv_buf[num_parts];
    MPI_Gatherv(send_up, nparts_up, PARTICLE, recv_buf, recvcounts,displacements, PARTICLE, 0,MPI_COMM_WORLD);

    if (rank==0){    
        for (int i=0; i<tot_size;++i){
            parts[recv_buf[i].id-1] = recv_buf[i];
        }    
    }
    nparts_up = 0;
//    MPI_Barrier(MPI_COMM_WORLD);
}