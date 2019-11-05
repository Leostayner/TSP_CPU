#include <iostream>
#include <math.h>
#include <string>
#include <vector>
#include <utility>
#include <omp.h>
#include <iomanip>
#include <chrono>
#include <algorithm>
#include <random>

#define MAX_ITER 300
using namespace std::chrono;

struct point{
  int id;
  double x;
  double y;
};

double dist(point &p1, point &p2){    
    return sqrt(pow(p1.x - p2.x, 2) +
                pow(p1.y - p2.y, 2));
}

double path_dist(std::vector<point> &points){
    double d = dist(points[0], points[points.size()-1]);

    for(unsigned int i=1; i<points.size(); ++i)
        d += dist(points[i-1], points[i]);
    
    return d;
}

double path_dist2(std::vector<int> &seq, std::vector<point> &points){
    double d = dist(points[seq[seq.size() - 1]], points[seq[0]]);

    for(unsigned int i=0; i<seq.size()-1; ++i)
        d += dist(points[seq[i]], points[seq[i+1]]);
    
    return d;
}

bool isIntersecting(point p1, point p2,
                    point q1, point q2){
                        
          return (((q1.x-p1.x)*(p2.y-p1.y) - (q1.y-p1.y) * (p2.x-p1.x)) * 
                  ((q2.x-p1.x)*(p2.y-p1.y) - (q2.y-p1.y) * (p2.x-p1.x)) < 0)
            &&
                 (((p1.x-q1.x)*(q2.y-q1.y) - (p1.y-q1.y) * (q2.x-q1.x)) * 
                  ((p2.x-q1.x)*(q2.y-q1.y) - (p2.y-q1.y) * (q2.x-q1.x)) < 0);
}

void branch_and_bound(std::vector<point> &points, 
             int idx, double curr_cost, std::vector<int> &curr_sol, 
             double &best_cost, std::vector<int> &best_seq, 
             std::vector<bool> &usado, int start = 0){
    
    if(curr_cost > best_cost)
        return;

    if(idx == points.size()){    
        curr_cost += dist(points[curr_sol[0]], points[curr_sol[curr_sol.size()-1]]);
        if (curr_cost > best_cost)
            return;
        
        #pragma omp critical
        {
            if (curr_cost <= best_cost){        
                best_seq  = curr_sol;
                best_cost = curr_cost;
                //std::cout << "best:"<< best_cost <<std::endl;
            }
        }
        return;
    }
    
    int max_iter = (start)? start+1: points.size();
    int id = omp_get_thread_num();
        
    for(int i=start; i<max_iter; ++i){
        if(!usado[i]){
            usado[i] = true;
            curr_sol[idx] = i;
            
            double new_cost = curr_cost + dist(points[curr_sol[idx-1]],
                                               points[curr_sol[idx]]);
            
            branch_and_bound(points, idx+1, new_cost, curr_sol, best_cost, best_seq, usado);
            
            usado[i] = false;
            curr_sol[idx] = -1;    
        }
    } 
    return;
}



void local_search(std::vector<point> &points, double &best_cost){
    
    double new_cost = 0;
    
    for(int i=0; i<points.size()-1; ++i){
        for(int j=i+1; j<points.size(); ++j){
            int last = (j+1 == points.size()-1)? 0 : j+1;
            if(isIntersecting(points[i], points[i+1], points[j], points[last])){
                auto tmp = points[i+1];
                points[i+1]  = points[j];
                points[j]    = tmp;
            }
        }
    }

    new_cost = path_dist(points);
    if (new_cost < best_cost){
        #pragma omp critical
        {
            if (new_cost < best_cost){        
                best_cost = new_cost;
                std::cout << "best:"<< best_cost <<std::endl;
            }
        }
    }
}


int main(){
    std::cout << std::fixed <<std::setprecision(5);
    int N; std::cin >> N;

    std::vector<point> points;
    for(int i=0; i<N; ++i){
        point pt;
        pt.id = i;
        
        std::cin >> pt.x; std::cin >> pt.y;
        points.push_back(pt);
    }

    std::vector<int> best_sol(N, -1);
    double best_cost = INFINITY;
    
    unsigned seed = system_clock::now().time_since_epoch().count();
    auto time_start = high_resolution_clock::now();

    #pragma omp parallel
    {
        #pragma omp master
        {   
            for(int i=0; i<MAX_ITER; ++i) {
                #pragma omp task shared(best_cost)
                {   
                    std::vector<point> local_points = points;
                    std::random_shuffle(local_points.begin()+1, local_points.end());
                    local_search(local_points, best_cost);
                }
            }
            
            for(int i=1; i<N; ++i) {
                #pragma omp task shared(best_cost, best_sol)
                {
                    std::vector<bool> usado(N, false);
                    std::vector<int> curr_sol(N, -1);
                    
                    curr_sol[0] = 0;
                    usado[0]    = true;
    
                    branch_and_bound(points, 1, 0, curr_sol, best_cost, best_sol, usado, i);
                }
            }
        }
    }

    auto time_end = duration_cast<duration<double>>(high_resolution_clock::now() - time_start).count();
       
    std::cout << path_dist2(best_sol, points) << " 1" << std::endl;
    
    for (auto e = best_sol.begin(); e != best_sol.end(); ++e)
        std::cout << *e << " ";
    
    std::cout << std::endl;
    std::cerr << "Time: " << time_end << std::endl;
}