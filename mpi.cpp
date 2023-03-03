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
static int num_procs_eff;

static int nranksx;
static int nranksy;

static int mybinx_min,mybinx_max; // index of min (included) and max (excluded) bin row
static int mybinx_min_g,mybinx_max_g; // index of min (included) and max (excluded) bin row, taking into account ghost bins
static int binxthresh_up,binxthresh_dn; // index of max bin row held by the processor above the current one, and min bin row held by the processor below

static int mybiny_min,mybiny_max; // index of min (included) and max (excluded) bin col
static int mybiny_min_g,mybiny_max_g; // index of min (included) and max (excluded) bin col, taking into account ghost bins
static int binythresh_rt,binythresh_lt; // index of max bin col held by the processor above the current one, and min bin col held by the processor below

static std::vector<std::vector<std::list<int>>> bins; //bins pertaining to one processor + ghost bins
static std::vector<std::vector<std::list<int>::iterator>> heads; // array of pointers to bin heads. This is used when moving particles between bins.

static int max_sent_parts;
static particle_t* send_up; //array of particles to be transferred up
static particle_t* send_dn; //array of particles to be transferred down
static int nparts_up;
static int nparts_dn;

static particle_t* send_rt; //array of particles to be transferred right
static particle_t* send_lt; //array of particles to be transferred left
static int nparts_rt;
static int nparts_lt;

static particle_t* send_ur; //array of particles to be transferred up right
static particle_t* send_ul; //array of particles to be transferred up left
static int nparts_ur;
static int nparts_ul;

static particle_t* send_dr; //array of particles to be transferred down right
static particle_t* send_dl; //array of particles to be transferred down left
static int nparts_dr;
static int nparts_dl;

void init_simulation(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
	// You can use this space to initialize data objects that you may need
	// This function will be called once before the algorithm begins
	// Do not do any particle simulation here    
    nbinsx = ((int)((double) size / (cutoff)) + 1);  // bin size greater or equal to cutoff length
    dxbin = size / (double) nbinsx;
    
    // Compute domain decomposition
    nranksx = (int) sqrt((double) num_procs);
    nranksy = num_procs / nranksx;
    nranksx = fmin(nranksx,nbinsx/5);
    nranksy = fmin(nranksy,nbinsx/5);
    if((num_procs==128) && (nbinsx/5 > 16)){nranksx = 16;nranksy = 8;} // optimize number of processors used for this case
    
    // Here, we make sure there are at least 5 bins/rank  - this is our heuristic to ensure particles don't jump between more than one rank per time step.
    num_procs_eff = nranksx*nranksy;
    if(rank==0) std::cout << "Using " << num_procs_eff << " processors" << std::endl;
    if(rank >= num_procs_eff) return;
    
    // This block performs some arithmetic to distribute rows of bins between processors
    int nbins_per_proc_x[nranksx]; // number of rows of bins held by each processor
    int nbins_per_proc_y[nranksy]; // number of rows of bins held by each processor
    int binxmin[nranksx+1],binxmax[nranksx]; // indices of min/max rows held by each processor
    int binymin[nranksy+1],binymax[nranksy]; // indices of min/max rows held by each processor
    binxmin[0] = 0;
    binymin[0] = 0;
    for (int p=0; p<nranksx;++p){
        nbins_per_proc_x[p] = (nbinsx + p) / nranksx; // arithmetic to make sure all bins are taken
        binxmin[p+1] = binxmin[p] + nbins_per_proc_x[p];
        binxmax[p] = fmin(nbinsx,binxmin[p+1]);
    }
    for (int p=0; p<nranksy;++p){
        nbins_per_proc_y[p] = (nbinsx + p) / nranksy; // arithmetic to make sure all bins are taken
        binymin[p+1] = binymin[p] + nbins_per_proc_y[p];
        binymax[p] = fmin(nbinsx,binymin[p+1]);
    }
    mybinx_min = binxmin[rank/nranksy];
    mybinx_max = binxmax[rank/nranksy];
    mybiny_min = binymin[rank%nranksy];
    mybiny_max = binymax[rank%nranksy];
    // add ghost bins
    mybinx_min_g = fmax(0,mybinx_min-1);
    mybinx_max_g = fmin(nbinsx,mybinx_max+1);
    mybiny_min_g = fmax(0,mybiny_min-1);
    mybiny_max_g = fmin(nbinsx,mybiny_max+1);
    
    // compute indices of rows held by neighboring processors -- to know when to send particles away
    binxthresh_up = (mybinx_min_g==0) ? -1 : mybinx_min-1;
    binxthresh_dn = (mybinx_max_g==nbinsx) ? nbinsx+1 : mybinx_max;
    binythresh_lt = (mybiny_min_g==0) ? -1 : mybiny_min-1;
    binythresh_rt = (mybiny_max_g==nbinsx) ? nbinsx+1 : mybiny_max;
        
    /// Allocate memory for bins in this rank only -- plus ghost bins. Also allocate memory for the heads. ///
    for (int ib = mybinx_min_g; ib < mybinx_max_g; ++ib) {
        std::vector<std::list<int>> row;
        bins.push_back(row);
        std::vector<std::list<int>::iterator> row2;
        heads.push_back(row2);
        for (int jb = mybiny_min_g; jb < mybiny_max_g; ++jb) {
            std::list<int> list = {};
            bins[ib-mybinx_min_g].push_back(list);
            std::list<int>::iterator hd = list.begin();
            heads[ib-mybinx_min_g].push_back(hd);
        }
    }
    
    max_sent_parts = 3*num_parts/num_procs_eff; // Heuristic for the maximum number of particles sent from one rank to another
    send_up = new particle_t[max_sent_parts] ; // buffers used to send particles between ranks
    send_dn = new particle_t[max_sent_parts] ; // buffers used to send particles between ranks
    nparts_up = 0;
    nparts_dn = 0;
    
    send_rt = new particle_t[max_sent_parts] ; // buffers used to send particles between ranks
    send_lt = new particle_t[max_sent_parts] ; // buffers used to send particles between ranks
    nparts_rt = 0;
    nparts_lt = 0;
    
    send_ur = new particle_t[max_sent_parts] ; // buffers used to send particles between ranks
    send_ul = new particle_t[max_sent_parts] ; // buffers used to send particles between ranks
    nparts_ur = 0;
    nparts_ul = 0;
    
    send_dr = new particle_t[max_sent_parts] ; // buffers used to send particles between ranks
    send_dl = new particle_t[max_sent_parts] ; // buffers used to send particles between ranks
    nparts_dr = 0;
    nparts_dl = 0;
    
//    if(rank==0) std::cout << "nbinsx:" << nbinsx  << "nranksx:" << nranksx << " nranksy:" << nranksy <<  std::endl;
//    std::cout << "rank" << rank << "binxthresh_up:" << binxthresh_up << " binxthresh_dn:" <<binxthresh_dn << "binythresh_lt:" << binythresh_lt << " binythresh_rt:" << binythresh_rt << std::endl;
               
    /// INITIALIZE - compute which bin each particle is in, and add it if the bin pertains to the current processor  ///
    for (int i = 0; i < num_parts; ++i) {
        int ib = (int)(parts[i].x / dxbin);
        int jb = (int)(parts[i].y / dxbin);
        if ((ib >= mybinx_min_g) && (ib < mybinx_max_g)){
            if ((jb >= mybiny_min_g) && (jb < mybiny_max_g)){
                bins[ib-mybinx_min_g][jb-mybiny_min_g].emplace_front(i);
            }
        }
    }
}







/* The following subroutine places particles that have been received from 
   other processors in their bins and in the "parts" array
*/
void emplace_particles(particle_t* parts, particle_t* recv_buf, int nparts){
    for (int i = 0; i < nparts; ++i){
        int parti = recv_buf[i].id-1; // For some reason the ids are defined as i+1 in main.cpp
        parts[parti] = recv_buf[i];
        int ib = (int)(recv_buf[i].x / dxbin);
        int jb = (int)(recv_buf[i].y / dxbin);
//        if ((ib >= mybinx_min) && (ib < mybinx_max)){ //this check shouldn't be needed
            bins[ib-mybinx_min_g][jb-mybiny_min_g].emplace_front(parti); 
//        }
    }
}



/* The following subroutine communicates particle buffers between processors 
   - up and down, right and left, and along diagonals. Those buffers are populated
   in simulate_one_step, twice: for particles that have moved away from one 
   processor's realm, and for ghost particles. The communication is made in two
   passes (down then up, or right then left, etc), to avoid deadlocks.
*/
void communicate_all(particle_t* parts, int rank){
    ////////////////////////////////////////////////
    /////////////////  UP AND DOWN /////////////////
    ////////////////////////////////////////////////
    particle_t recv_above[max_sent_parts];
    int nparts_above = 0;
    particle_t recv_below[max_sent_parts];
    int nparts_below = 0;
    MPI_Status status;
    
    if (mybinx_max_g<nbinsx){
        MPI_Send(send_dn, nparts_dn, PARTICLE, rank+nranksy, 0, MPI_COMM_WORLD);
    }

    if (mybinx_min_g > 0){
        MPI_Recv(recv_above, max_sent_parts, PARTICLE, rank-nranksy, 0, MPI_COMM_WORLD,&status);     
        MPI_Get_count(&status, PARTICLE, &nparts_above);
        
        MPI_Send(send_up, nparts_up, PARTICLE, rank-nranksy, 1, MPI_COMM_WORLD);
    }

    if (mybinx_max_g<nbinsx){        
        MPI_Recv(recv_below, max_sent_parts, PARTICLE, rank+nranksy, 1, MPI_COMM_WORLD,&status);
        MPI_Get_count(&status, PARTICLE, &nparts_below);
    }
    
    // Emplace received particles
    emplace_particles(parts,recv_above, nparts_above);
    emplace_particles(parts,recv_below, nparts_below);
    
    ////////////////////////////////////////////////
    /////////////////  RIGHT AND LEFT //////////////
    ////////////////////////////////////////////////
    nparts_above = 0;
    nparts_below = 0;
    
    if (mybiny_max_g<nbinsx){
        MPI_Send(send_rt, nparts_rt, PARTICLE, rank+1, 2, MPI_COMM_WORLD);
    }

    if (mybiny_min_g > 0){
        MPI_Recv(recv_above, max_sent_parts, PARTICLE, rank-1, 2, MPI_COMM_WORLD,&status);     
        MPI_Get_count(&status, PARTICLE, &nparts_above);
        
        MPI_Send(send_lt, nparts_lt, PARTICLE, rank-1, 3, MPI_COMM_WORLD);
    }

    if (mybiny_max_g<nbinsx){        
        MPI_Recv(recv_below, max_sent_parts, PARTICLE, rank+1, 3, MPI_COMM_WORLD,&status);
        MPI_Get_count(&status, PARTICLE, &nparts_below);
    } 

    // Emplace received particles
    emplace_particles(parts,recv_above, nparts_above);
    emplace_particles(parts,recv_below, nparts_below);
    
    //////////////////////////////////////////////////////////////
    /////////////////  UPPER RIGHT AND DOWN LEFT /////////////////
    //////////////////////////////////////////////////////////////
    nparts_above = 0;
    nparts_below = 0;
    
    if ((mybinx_min_g>0)&&(mybiny_max_g<nbinsx)){
        MPI_Send(send_ur, nparts_ur, PARTICLE, rank+1-nranksy, 4, MPI_COMM_WORLD);
    }

    if ((mybiny_min_g > 0)&&(mybinx_max_g<nbinsx)){
        MPI_Recv(recv_above, max_sent_parts, PARTICLE, rank-1+nranksy, 4, MPI_COMM_WORLD,&status);     
        MPI_Get_count(&status, PARTICLE, &nparts_above);
        
        MPI_Send(send_dl, nparts_dl, PARTICLE, rank-1+nranksy, 5, MPI_COMM_WORLD);
    }

    if ((mybinx_min_g>0)&&(mybiny_max_g<nbinsx)){        
        MPI_Recv(recv_below, max_sent_parts, PARTICLE, rank+1-nranksy, 5, MPI_COMM_WORLD,&status);
        MPI_Get_count(&status, PARTICLE, &nparts_below);
    }
    // Emplace received particles
    emplace_particles(parts,recv_above, nparts_above);
    emplace_particles(parts,recv_below, nparts_below);
    
    //////////////////////////////////////////////////////////////
    /////////////////  DOWN RIGHT AND UPPER LEFT /////////////////
    //////////////////////////////////////////////////////////////
    nparts_above = 0;
    nparts_below = 0;
    
    if ((mybinx_min_g>0)&&(mybiny_min_g > 0)){
        MPI_Send(send_ul, nparts_ul, PARTICLE, rank-1-nranksy, 6, MPI_COMM_WORLD);
    }

    if ((mybinx_max_g<nbinsx)&&(mybiny_max_g<nbinsx)){
        MPI_Recv(recv_above, max_sent_parts, PARTICLE, rank+1+nranksy, 6, MPI_COMM_WORLD,&status);     
        MPI_Get_count(&status, PARTICLE, &nparts_above);
        
        MPI_Send(send_dr, nparts_dr, PARTICLE, rank+1+nranksy, 7, MPI_COMM_WORLD);
    }

    if ((mybinx_min_g>0)&&(mybiny_min_g>0)){        
        MPI_Recv(recv_below, max_sent_parts, PARTICLE, rank-1-nranksy, 7, MPI_COMM_WORLD,&status);
        MPI_Get_count(&status, PARTICLE, &nparts_below);
    }
    // Emplace received particles
    emplace_particles(parts,recv_above, nparts_above);
    emplace_particles(parts,recv_below, nparts_below);
    
}

    
    
    
    
    
    
    


void simulate_one_step(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    if(rank >= num_procs_eff) return; // For unused processors
    //////////////////////////////////////////////
    //// Compute Forces -- exclude ghost bins ////
    //////////////////////////////////////////////
    for (int ib = mybinx_min; ib < mybinx_max; ++ib) {
        for (int jb = mybiny_min; jb < mybiny_max; ++jb) {
            // Keep track of list heads for the next step
            heads[ib-mybinx_min_g][jb-mybiny_min_g] = bins[ib-mybinx_min_g][jb-mybiny_min_g].begin();
            for (int i : bins[ib-mybinx_min_g][jb-mybiny_min_g]) {
                parts[i].ax = parts[i].ay = 0;
                for (int local_i = fmax(0,ib-1); local_i <= fmin(nbinsx-1,ib+1); ++local_i){
                    for (int local_j = fmax(0,jb-1); local_j <= fmin(nbinsx-1,jb+1); ++local_j){
                        for (int j : bins[local_i-mybinx_min_g][local_j-mybiny_min_g]) {
                                apply_force(parts[i], parts[j]);
                        }
                    }
                }
            }
        }
    }
    

    ///////////////////////////////////////////////////////////////////////////////////////////////
    //// Move Particles & keep track of particles that move out of the scope of this processor ////
    ///////////////////////////////////////////////////////////////////////////////////////////////
    for (int ib = mybinx_min; ib < mybinx_max; ++ib) {
        for (int jb = mybiny_min; jb < mybiny_max; ++jb) {
            // Iterate over particles. Because particles that have moved from bins already treated 
            // may have been added to the current bin, we kept a pointer to the original list head.
            std::list<int>::iterator it = heads[ib-mybinx_min_g][jb-mybiny_min_g];
            while (it != bins[ib-mybinx_min_g][jb-mybiny_min_g].end()){
                int i = *it;
                move(parts[i], size);
                int new_ib = (int)(parts[i].x / dxbin);
                int new_jb = (int)(parts[i].y / dxbin);
                
                if (ib != new_ib || jb != new_jb) {
                    it = bins[ib-mybinx_min_g][jb-mybiny_min_g].erase( it );
                    if ((new_ib >= mybinx_min) && (new_ib < mybinx_max) && (new_jb >= mybiny_min) && (new_jb < mybiny_max)){
                        bins[new_ib-mybinx_min_g][new_jb-mybiny_min_g].emplace_front(i);
                    }
                    if ((new_ib <= binxthresh_up) && (new_jb >= mybiny_min) && (new_jb < mybiny_max)) { // add to array to be communicated to processor above
                        send_up[nparts_up] = parts[i];
                        nparts_up++;
                    }
                    else if ((new_ib >= binxthresh_dn) && (new_jb >= mybiny_min) && (new_jb < mybiny_max)) { // add to array to be communicated to processor above
                        send_dn[nparts_dn] = parts[i];
                        nparts_dn++;
                    }
                    else if ((new_jb <= binythresh_lt) && (new_ib >= mybinx_min) && (new_ib < mybinx_max)) { // add to array to be communicated to processor left
                        send_lt[nparts_lt] = parts[i];
                        nparts_lt++;
                    }
                    else if ((new_jb >= binythresh_rt) && (new_ib >= mybinx_min) && (new_ib < mybinx_max)) { // add to array to be communicated to processor right
                        send_rt[nparts_rt] = parts[i];
                        nparts_rt++;
                    }
                    else if ((new_ib <= binxthresh_up) && (new_jb <= binythresh_lt)) { // etc with corners
                        send_ul[nparts_ul] = parts[i];
                        nparts_ul++;
                    }
                    else if ((new_ib <= binxthresh_up) && (new_jb >= binythresh_rt)) { // etc with corners
                        send_ur[nparts_ur] = parts[i];
                        nparts_ur++;
                    }
                    else if ((new_ib >= binxthresh_dn) && (new_jb <= binythresh_lt)) { // etc with corners
                        send_dl[nparts_dl] = parts[i];
                        nparts_dl++;
                    }
                    else if ((new_ib >= binxthresh_dn) && (new_jb >= binythresh_rt)) { // etc with corners
                        send_dr[nparts_dr] = parts[i];
                        nparts_dr++;
                    }
                }
                else{
                    it ++;
                }
            }
        }
    }

    ////////////////////////////////////////////////
    //// Communicate particles that have moved  ////
    ////////////////////////////////////////////////
    
    communicate_all(parts,rank);

    ////////////////////////////////////////////////////
    //// Populate ghost bins buffers before sending ////
    ////////////////////////////////////////////////////
    nparts_up = 0;
    nparts_dn = 0; 
    nparts_rt = 0;
    nparts_lt = 0;
    nparts_ur = 0;
    nparts_ul = 0;
    nparts_dr = 0;
    nparts_dl = 0;
       
    // bottom ghost bin
    if (mybinx_max_g<nbinsx){
        for (int jb = mybiny_min; jb < mybiny_max; ++jb) {
            for (int i : bins[mybinx_max-1-mybinx_min_g][jb-mybiny_min_g]) {
                send_dn[nparts_dn] = parts[i];
                nparts_dn++;
            }
        }
    }
    // top ghost bin
    if (mybinx_min_g > 0){
        for (int jb = mybiny_min; jb < mybiny_max; ++jb) {
            for (int i : bins[1][jb-mybiny_min_g]) {
                send_up[nparts_up] = parts[i];
                nparts_up++;
            }
        }
    }
    // right ghost bin
    if (mybiny_max_g<nbinsx){
        for (int ib = mybinx_min; ib < mybinx_max; ++ib) {
            for (int i : bins[ib-mybinx_min_g][mybiny_max-1-mybiny_min_g]) {
                send_rt[nparts_rt] = parts[i];
                nparts_rt++;
            }
        }
    }
    // left ghost bin
    if (mybiny_min_g > 0){
        for (int ib = mybinx_min; ib < mybinx_max; ++ib) {
            for (int i : bins[ib-mybinx_min_g][1]) {
                send_lt[nparts_lt] = parts[i];
                nparts_lt++;
            }
        }
    }
    // upper right ghost bin
    if ((mybinx_min_g>0)&&(mybiny_max_g<nbinsx)){
        for (int i : bins[1][mybiny_max-1-mybiny_min_g]) {
            send_ur[nparts_ur] = parts[i];
            nparts_ur++;
        }
    }
    // lower left ghost bin
    if ((mybinx_max_g<nbinsx)&&(mybiny_min_g>0)){
        for (int i : bins[mybinx_max-1-mybinx_min_g][1]) {
            send_dl[nparts_dl] = parts[i];
            nparts_dl++;
        }
    }    
    // upper left ghost bin
    if ((mybinx_min_g>0)&&(mybiny_min_g>0)){
        for (int i : bins[1][1]) {
            send_ul[nparts_ul] = parts[i];
            nparts_ul++;
        }
    }    
    // lower right ghost bin
    if ((mybinx_max_g<nbinsx)&&(mybiny_max_g<nbinsx)){
        for (int i : bins[mybinx_max-1-mybinx_min_g][mybiny_max-1-mybiny_min_g]) {
            send_dr[nparts_dr] = parts[i];
            nparts_dr++;
        }
    }  
    

    ////// EMPTY GHOST BINS/////
    if (mybinx_max_g<nbinsx){
        for (int jb = mybiny_min_g; jb < mybiny_max_g; ++jb) {
            bins[mybinx_max_g-1-mybinx_min_g][jb-mybiny_min_g].clear();
        }
    }
    if (mybinx_min_g > 0){
        for (int jb = mybiny_min_g; jb < mybiny_max_g; ++jb) {
            bins[0][jb-mybiny_min_g].clear();
        }
    }
    if (mybiny_max_g<nbinsx){
        for (int ib = mybinx_min_g; ib < mybinx_max_g; ++ib) {
            bins[ib-mybinx_min_g][mybiny_max_g-1-mybiny_min_g].clear();
        }
    }
    if (mybiny_min_g > 0){
        for (int ib = mybinx_min_g; ib < mybinx_max_g; ++ib) {
            bins[ib-mybinx_min_g][0].clear();
        }
    }

    /////////////////////////////////
    //// Communicate ghost bins  ////
    /////////////////////////////////
    
    communicate_all(parts,rank);

    // Prepare for next step - Reset the buffer counters for send_up and send_dn to 0
    nparts_up = 0;
    nparts_dn = 0; 
    nparts_rt = 0;
    nparts_lt = 0;
    nparts_ur = 0;
    nparts_ul = 0;
    nparts_dr = 0;
    nparts_dl = 0;   
    
}









void gather_for_save(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    // Write this function such that at the end of it, the master (rank == 0)
    // processor has an in-order view of all particles. That is, the array
    // parts is complete and sorted by particle id.
    
    // Count particles in the current rank and gather them in the send_up buffer
    nparts_up=0;
    if(rank < num_procs_eff){
        for (int ib = mybinx_min; ib < mybinx_max; ++ib) {
            for (int jb = mybiny_min; jb < mybiny_max; ++jb) {
                for (int i : bins[ib-mybinx_min_g][jb-mybiny_min_g]) {
                    send_up[nparts_up] = parts[i];
                    nparts_up++;
                }
            }
        }
    }

    // Gather number of particles that will get sent at the next step (for MPI_Gatherv)
    int recvcounts[num_procs];
    MPI_Gather(&nparts_up, 1, MPI_INT, recvcounts, 1, MPI_INT, 0,MPI_COMM_WORLD);
    
    // Make array of starting indices for the receiving buffer
    int displacements[num_procs];
    displacements[0]=0;
    for (int p=1; p<num_procs;++p){
        displacements[p] = displacements[p-1] + recvcounts[p-1];
    }
    int tot_size = displacements[num_procs-1] + recvcounts[num_procs-1]; // should be == num_parts

    // Gatherv all the particles to rank 0
    particle_t recv_buf[num_parts];
    MPI_Gatherv(send_up, nparts_up, PARTICLE, recv_buf, recvcounts,displacements, PARTICLE, 0,MPI_COMM_WORLD);

    // Emplace all the particles in the 'parts' array
    if (rank==0){    
        for (int i=0; i<tot_size;++i){
            parts[recv_buf[i].id-1] = recv_buf[i];
        }    
    }
    
    // Reset the send_up buffer counter to 0
    nparts_up = 0;
}