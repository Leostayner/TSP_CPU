//g++ tsp-seq.cpp -o tsp-seq -fopenmp -O3
#include <iostream>
#include <math.h>
#include <string>
#include <vector>
#include <utility>
#include <omp.h>
#include <iomanip>
#include <chrono>

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

double path_dist(std::vector<int> &seq, std::vector<point> &points){
    double d = dist(points[seq[seq.size() - 1]], points[seq[0]]);

    for(unsigned int i=0; i<seq.size()-1; ++i)
        d += dist(points[seq[i]], points[seq[i+1]]);
    
    return d;
}

double backtrack(std::vector<point> &points, 
             int idx, double curr_cost, std::vector<int> &curr_sol, 
             double best_cost, std::vector<int> &best_seq, 
             std::vector<bool> &usado){
    
    if(idx == points.size()){    
        curr_cost += dist(points[curr_sol[0]], points[curr_sol[curr_sol.size()-1]]);
        if (curr_cost < best_cost){
            best_seq  = curr_sol;
            best_cost = curr_cost;
            //std::cout << "best:"<< best_cost <<std::endl;
        }
        return best_cost;
    }
    

    for(int i=0; i<points.size(); ++i){
        if(!usado[i]){
            usado[i] = true;
            curr_sol[idx] = i;
            
            double new_cost = curr_cost + dist(points[curr_sol[idx-1]],
                                               points[curr_sol[idx]]);
             
            best_cost = backtrack(points, idx+1, new_cost, curr_sol, best_cost, best_seq, usado);
            
            usado[i] = false;
            curr_sol[idx] = -1;    
        }
    } 
    return best_cost;
}


int main(){
    std::cout << std::fixed <<std::setprecision(5);

    int N; std::cin >> N;
    std::vector<point> points;
    
    for(int i=0; i<N; ++i){
        point point;
        std::cin >> point.x; std::cin >> point.y;
        points.push_back(point);
    }

    std::vector<bool> usado(N, false);
    std::vector<int> curr_sol(N, -1);
    std::vector<int> best_sol(N, -1);

    curr_sol[0] = 0;
    usado[0]    = true;
    
    auto time_start = high_resolution_clock::now();
    backtrack(points, 1, 0, curr_sol, INFINITY, best_sol, usado);
    auto time_end = duration_cast<duration<double>>(high_resolution_clock::now() - time_start).count();

    std::cout << path_dist(best_sol, points) << " 1" << std::endl;
    for (auto e = best_sol.begin(); e != best_sol.end(); ++e){
        if(e != best_sol.end() -1)
            std::cout << *e << " ";
        else
            std::cout << *e << std::endl;;
    }
    std::cerr << "Time: " << time_end << std::endl; 
}