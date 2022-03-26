// By: Paras Savnani
/** Citations: 
1) https://en.wikipedia.org/wiki/A*_search_algorithm
2) https://i11www.iti.kit.edu/extra/publications/bdgwz-sfpcs-15.pdf
3)  https://www.sciencedirect.com/science/article/abs/pii/0377221782902053
*/
// Compiler String: g++ -std=c++11 -O1 main.cpp network.cpp -o solution 

#include <cmath>
#include <climits>
#include <limits>
#include <vector>
#include <queue>
#include <unordered_map>
#include "network.h"

#define th2r(x) (x*3.141592653589/180.0)


/**
 * @brief Node Class 
 * A class for extending the row struct to include battery capaity and define new type of node. 
 * It also has an == operator overloding which is helpful to check equality of the class instances for hash collisions
 */
class Node {
public:
    row value;              // struct having name,lat,lon,rate information of the node
    int capacity;           // charge cpacity time remaining on the battery 
    
    Node(){}
    Node(row value, int capacity){
        this->value = value;
        this->capacity = capacity;         
    }

    // For Node equality, we are checking only name from row object because same name row object will have same lat, lon and rate.
    bool operator==(const Node& p) const{
        return (value.name == p.value.name) && (capacity == p.capacity);
    }
};

/**
 * @brief Custom Hash Function Class 
 * A class for defining the hash function for the unordered_map container with Node as key in it.
 * We define our hash function as XOR of the predefined string and int hash values.
 */

class HashFunction {
public:
    size_t operator()(const Node& p) const{
        return (std::hash<std::string>()(p.value.name)) ^ (std::hash<int>()(p.capacity));
    }
};

/**
 * @brief Graph Class having all the utilities like the genration and search functions for the program.
 * A class for defining the main graph data structure for the challenge. It has all the required attributes and memebers 
 * to generate and search the graph efficiently. 
 */

class Graph {
private:
    int n;                                                          //no. of charging stations
    float max_capacity;                                             //Max distance on full charge
    float velocity;                                                 //Car's travelling velocity
    float radius;                                                   //Earth's Radius
    float divisions;                                                //No of divisions for battery capacity
    struct CompareValue;                                            //Custom struct to compare values in priority queue
    std::vector<std::vector<float>> DistanceMatrix;                           //Great Circle distances between charging stations

public:
    std::unordered_map<Node, std::vector<Node>, HashFunction> adjlist;                             //Adjacency list of graph
    Graph(int n, float max_capacity, float velcity, float radius, float divisions);      //Graph constructor
    void generateDistanceMatrix();                                                       //Fucntion to generate Time Matrix
    void generateGraph();                                                                //Fucntion to generate Graph Adjacency list
    void backtrack(std::unordered_map<Node, Node, HashFunction> cameFrom, Node current);      //Backtracking function to find the shortest path once goal is reached
    float heuristic(Node start, Node goal);                                              //A star search heuristic function
    void astarSearch(Node start, Node goal);                                             //Optimized A Star search function
};

/**
 * @brief Initializes the Graph class attributes
 *
 * @param n no. of charging stations
 * @param max_capacity Max distance that car can travel on full charge
 * @param velocity Car's travelling velocity
 * @param radius Earth's Radius
 * @param divisions No of divisions for battery capacity
 */
Graph::Graph(int n, float max_capacity, float velocity, float radius, float divisions)
{
    this->n = n;
    this->max_capacity = max_capacity;
    this->velocity = velocity;
    this->radius = radius;
    this->divisions = divisions;
    std::vector<std::vector<float>> DistanceMatrix(n, std::vector<float>(n, 0.0));
    this->DistanceMatrix = DistanceMatrix;
}


/**
 * @brief Function responsible to generate the distance Matrix that will be used to retrive the 
 *        greatcircle distances between coordinates.
 */
void Graph::generateDistanceMatrix()
{
    // Can be optimized to store only upper triangular matrix
    float lat; float lon; float greatCircleDistance;
    for(int i=0; i<n; i++)
    {
        for(int j=0; j<n; j++)
        {
            // Haversine distance
            greatCircleDistance = radius*2.0*asin(sqrt(pow(sin(th2r((network[i].lat-network[j].lat)/2.)),2) + 
                                    cos(th2r(network[i].lat))*cos(th2r(network[j].lat))* pow(sin(th2r((network[i].lon-network[j].lon)/2.)),2)));

            if(greatCircleDistance > 0.1)
                DistanceMatrix[i][j] = greatCircleDistance;
            else
                DistanceMatrix[i][j] = 0.0;
        }
    }
}

/**
 * @brief This function is responsible to create the main graph by dividing each charging station into sub-divisions and
 *        connecting each node with the corresponding nodes by taking travel distance and charging capacity into account.
 */

void Graph::generateGraph()
{
    generateDistanceMatrix();

    // Build Graph
    for(int i=0; i<n; i++)
    {
        float rate_i =  network[i].rate;
        float time_for_each_div_i = (max_capacity/rate_i)/divisions;    //in hrs 

        for(int j=0; j<=divisions; j++)
        {
            Node new_node = Node(network[i], j);

            if(j!=divisions)
                adjlist[new_node].push_back(Node(network[i], j+1));

            for(int k=0; k<n; k++)
            {
                float rate_k = network[k].rate;
                float time_for_each_div_k = (max_capacity/rate_k)/divisions;    //in hrs
                float distance_to_travel = DistanceMatrix[i][k];
                float distance_can_travel = j*time_for_each_div_i*rate_i;
                if((distance_to_travel > 0) && (distance_to_travel < distance_can_travel))
                {
                    float rem_distance = distance_can_travel-distance_to_travel;
                    int new_div = rem_distance/(rate_k*time_for_each_div_k);
                    adjlist[new_node].push_back(Node(network[k], new_div));
                }
            }
        }
    }
}

/**
 * @brief Backtracking function to print the path from start to goal,
 *        once the goal is reached.
 * @param cameFrom map of parent child nodes for backtracking
 * @param current goal node 
 */
void Graph::backtrack(std::unordered_map<Node, Node, HashFunction> cameFrom, Node current)
{
    std::vector<Node> path;
    path.push_back(current);
    while(cameFrom.find(current)!=cameFrom.end())
    {
        current = cameFrom[current];
        path.insert(path.begin(), current);
    }
    
    std::cout << path[0].value.name << ", ";
    float time_for_each_div; int count=0;
    Node last_node = path[1];

    for (auto it = begin(path)+1; it != end(path); ++it) 
    {
        if(last_node.value.name!=it->value.name)
        {
            time_for_each_div = (max_capacity/last_node.value.rate)/divisions;
            std::cout << last_node.value.name << ", " <<  count*time_for_each_div << ", ";
            count=0;
        }
        count++;
        last_node = *it;
    }
    std::cout << path[path.size()-1].value.name << std::endl;
}

/**
 * @brief Heuristic function for the A star search
 * @param start start node
 * @param goal goal node 
 * @return time required to travel from start to goal
 */

float Graph::heuristic(Node start, Node goal)
{
    // Time to reach Goal using haversine distance 
    float greatCircleDistance = radius*2.0*asin(sqrt(pow(sin(th2r((start.value.lat-goal.value.lat)/2.)),2) + 
                                    cos(th2r(start.value.lat))*cos(th2r(goal.value.lat))* pow(sin(th2r((start.value.lon-goal.value.lon)/2.)),2)));
    float time = (greatCircleDistance/velocity);   // in hrs  
    return time; 
}

/**
 * @brief This structure implements the operator overloading for the custom prioirty queue values.
 */
struct Graph::CompareValue {
    bool operator()(std::pair<float, Node> const& p1, std::pair<float, Node> const& p2) const{
        return p1.first > p2.first;
    }
};


/**
 * @brief A Star Shortest Path Search ALgorithm: f(n)(cost) = g(n)(cost to come from source) + h(n)(cost to go to goal) 
 * @param start start node
 * @param goal goal node 
 */
void Graph::astarSearch(Node start, Node goal)
{
    // Check if goal is reachable without charging the car between travel
    for(int i=0; i<n; i++)
    {
        for(int j=0; j<n; j++)
        {
            if(network[i].name==start.value.name && network[j].name==goal.value.name && DistanceMatrix[i][j] <= 320)
            {
                std::cout << start.value.name << ", " << goal.value.name << std::endl;
                return;
            }
        }
    }

    // Cost will be time (time to come to this node + time to reach destination)
    // Min heap of (fScore, Node)
    std::priority_queue<std::pair<float, Node>, std::vector<std::pair<float, Node>>, CompareValue> pq;   

    // For node n, cameFrom[n] is the node immediately preceding it on the cheapest path from start
    // to n currently known.
    std::unordered_map<Node, Node, HashFunction> cameFrom;
    
    std::unordered_map<Node, float, HashFunction> gScore;
    for(auto element:adjlist)
        gScore[element.first] = std::numeric_limits<float>::max();
        
    // Initilaize 
    gScore[start] = 0.0;  // in hrs
    float fScore_start = gScore[start] + heuristic(start, goal);
    pq.push({fScore_start, start});

    while(!pq.empty())
    {   
        auto current = pq.top().second;   // Current Node
        float time_for_each_div_curr = (max_capacity/current.value.rate)/divisions;    //in hrs

        if(current.value.name==goal.value.name && current.capacity==goal.capacity)
        {
            backtrack(cameFrom, current);
            return;
        } 

        pq.pop();   
        for(auto neighbour:adjlist[current])
        {
            // current.time is the time of the edge from current to neighbor
            // tentative_gScore is the time from start to the neighbor through current
            float time_for_each_div_neighbor = (max_capacity/neighbour.value.rate)/divisions;    //in hrs
            float rem_distance = neighbour.capacity*time_for_each_div_neighbor*neighbour.value.rate;
            float distance_covered = max_capacity-rem_distance;
            float tentative_gScore = gScore[current] + distance_covered/velocity;

            if(tentative_gScore < gScore[neighbour])
            {
                cameFrom[neighbour] = current;
                gScore[neighbour] = tentative_gScore;
                float fScore_neighbour = tentative_gScore + heuristic(neighbour, goal);
                pq.push({fScore_neighbour, neighbour});
            }
        }
    }
    std::cout << "Failure" << std::endl;
    return;
}


int main(int argc, char** argv)
{
    if (argc != 3)
    {
        std::cout << "Error: requires initial and final supercharger names" << std::endl;        
        return -1;
    }
    
    std::string initial_charger_name = argv[1];
    std::string goal_charger_name = argv[2];

    if(initial_charger_name==goal_charger_name)
    {
        std::cout << "Same Start and Goal Locations";
        return 0;
    }

    int n = 303;                                
    float max_capacity = 320;                     
    float velocity = 105;
    float radius = 6356.752;                         
    float divisions = 100;       
    
    Graph graph(n, max_capacity, velocity, radius, divisions);

    // Generate Graph 
    graph.generateGraph();
    
    
    // Initializing Start, Goal Nodes and gScore vector
    Node start, goal;
    int min_cap=INT_MAX, max_cap=INT_MIN; 
    for(auto element:graph.adjlist)
    {
        if(element.first.value.name==initial_charger_name)
        {
            max_cap = std::max(max_cap, element.first.capacity);
            start = Node(element.first.value, max_cap);
        }
        if(element.first.value.name==goal_charger_name)
        {
            min_cap = std::min(min_cap, element.first.capacity);
            goal = Node(element.first.value, min_cap);
        }
    }

    // Search Graph 
    graph.astarSearch(start, goal);
    return 0;
}
