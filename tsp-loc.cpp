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

#define MAX_ITER 10000
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


bool isIntersecting(point p1, point p2,
                    point q1, point q2){
                        
          return (((q1.x-p1.x)*(p2.y-p1.y) - (q1.y-p1.y) * (p2.x-p1.x)) * 
                  ((q2.x-p1.x)*(p2.y-p1.y) - (q2.y-p1.y) * (p2.x-p1.x)) < 0)
            &&
                 (((p1.x-q1.x)*(q2.y-q1.y) - (p1.y-q1.y) * (q2.x-q1.x)) * 
                  ((p2.x-q1.x)*(q2.y-q1.y) - (p2.y-q1.y) * (q2.x-q1.x)) < 0);
}


void local_search(std::vector<point> &points, 
                  std::vector<point> &best_sol, double &best_cost){
    
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
                best_sol  = points;
                //std::cout << "best:"<< best_cost <<std::endl;
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

    std::vector<point> best_sol;
    double best_cost = INFINITY;

    unsigned seed = system_clock::now().time_since_epoch().count();
    auto time_start = high_resolution_clock::now();
    
    #pragma omp parallel
    {
        #pragma omp master
        {   
            for(int i=0; i<MAX_ITER; ++i) {
                #pragma omp task shared(best_sol, best_cost)
                {   
                    std::vector<point> local_points = points;
                    std::random_shuffle(local_points.begin()+1, local_points.end());
                    local_search(local_points, best_sol, best_cost);
                }
            }
        }
    }

    auto time_end = duration_cast<duration<double>>(high_resolution_clock::now() - time_start).count();
       
    std::cout << path_dist(best_sol) << " 0" << std::endl;

    for(int i=0; i< best_sol.size();++i)
        std::cout << best_sol[i].id << " "; 

    std::cout << std::endl;
    std::cerr << "Time: " << time_end << std::endl; 
}