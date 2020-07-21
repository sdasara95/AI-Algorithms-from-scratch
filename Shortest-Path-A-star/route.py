#!/bin/env python3

#----------------------------------------------------------------------------------------------------------------------------------
# by: Satyaraja Dasara
#
#(1) Description of formulation of search problem :
#
# I have considered each record in the given data file as a class object containing the start city, destination city, 
# distance, speedlimit, highway name, latitude and longitude values of the start city as the class attributes.
# I have created a dictionary where the key value is the city name and the corresponding key's value is a list of class 
# objects having their start city attribute same as the key value. This is my STATE SPACE which I used for each search 
# algorithm.
# The Successor method uses this dictionary to get the successors states from the destination city class attribute
# The edge weights based on the cost function given as input is computed in the respective search method without calling
# any additional methods using the route traversed so far which includes the start city and the intermediatary cities till
# the current state.
# The Goal State method checks whether the first and last string item in the entire route traversed string as the same as 
# the input start city and end city.
# I calculated the average speedlimit and assigned it to those cases where the speedlimit was zero.
# The heuristic I considered for the 'Distance' cost was the latitude longitude Haversine distance between two cities. 
# This is admissable heuristic because it gives the lowest distance between two cities.
# The heuristic I considered for the 'Time' cost was the 'Distance' heuristic divided by the average speed limit. 
# This is admissable because it never overestimates the time required to travel between two places.
# The heuristic I considered for the 'Segments' cost was assignment of value 0 if the destination city attribute in any 
# of the successors was the end city else 1.
#------------------------------------------------------------------------------------------------------------------------------
#
#(2) Working of search algorithm :
#
# Only A-star and uniform cost search incorporate the cost functions given as input in this code. IDS BFS and DFS don't take
# edge weights into consideration while finding the path hence the chosen cost function isn't passed to these methods. IDS 
# has been looped till infinity so it keeps on executing in case of no solution.
# BFS and DFS differ only in the way the successors are added to the fringe i.e. beginning or end.
# A closed list containing the visited nodes is being maintained in the above cases to prevent an infinite loop.
# Uniform Cost search has been implemented with a priority queue and works by considering the lowest cost of the successors.
# Astar has been implemented for all the cost functions with a priority queue.
#----------------------------------------------------------------------------------------------------------------------------------------
#
#(3) Assumptions, issues and design decisions:
#
# For the records having no speed limit i.e. speed limit as 0, I calculated the average speed limit for all the paths and
# assigned that value to those records where the speed limit was zero.
# For the cities that didn't have values of latitude and longitude in the city-gps file, I assigned them the corresponding
# values of their nearest neighbors.
# I used the city gps file to calculate the Haversine distance that I used as the heuristic distance. if t
# Astar and uniform cost search are performing well for long paths while BFS performs well for short paths. Astar can be improved
# if the inconsistency in data can be handled.
#-------------------------------------------------------------------------------------------------------------------------------



import os
import sys
from queue import PriorityQueue
from math import radians, cos, sin, asin, sqrt

#https://en.wikipedia.org/wiki/Haversine_formula
#https://stackoverflow.com/questions/4913349/haversine-formula-in-python-bearing-and-distance-between-two-gps-points
def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(lambda x: float(x), [lon1,lat1,lon2,lat2])
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula 
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 3956  # Radius of earth in kilometers. Use 6371 for kilometers
    return c * r

class City():

    def __init__(self,name,dest_city,dist,speedlim,highway):
        self.name = name
        self.dest_city = dest_city
        self.dist = dist
        self.speedlim = speedlim
        self.highway = highway

        self.latitude = 0
        self.longitude = 0

    def set_gps(self, latitude, longitude):
        self.latitude = latitude
        self.longitude = longitude



global list_classes
list_classes = {}
global avg_speedlim
avg_speedlim = 0
count=0
with open('road-segments.txt','r') as input_file:
    for line in input_file:
        record = line.split()
        if len(record)<5:
            origin_city,dest_city,dist,speedlim,highway = record[0],record[1],int(record[2]),0,record[3]
        else:
            origin_city,dest_city,dist,speedlim,highway = record[0],record[1],int(record[2]),int(record[3]),record[4]

        avg_speedlim += speedlim
        count+=1

        try:
            list_classes[origin_city] =  list_classes[origin_city]+[City(origin_city,dest_city,dist,speedlim,highway)]

        except:
            list_classes[origin_city] = [City(origin_city,dest_city,dist,speedlim,highway)]


        try:
            list_classes[dest_city] = list_classes[dest_city]+[City(dest_city,origin_city,dist,speedlim,highway)]

        except:
            list_classes[dest_city] = [City(dest_city,origin_city,dist,speedlim,highway)]


with open('city-gps.txt','r') as input_file:
    for line in input_file:
        record = line.split()
        city,latitude,longitude=record[0],record[1],record[2]
        if city in list(list_classes.keys()):
            for classes in list_classes[city]:
                classes.set_gps(latitude,longitude)


avg_speedlim/=count
avg_speedlim=int(avg_speedlim)

for i in list_classes.values():
    for j in i:
        if j.speedlim==0:
            j.speedlim=avg_speedlim

with open('city-gps.txt','r') as input_file:
    for line in input_file:
        record = line.split()
        city = record[0]
        if city in list(list_classes.keys()):
            for i in list_classes[city]:
                i.set_gps(record[1], record[2])

for i in list_classes:
    if list_classes[i][0].latitude==0 and list_classes[i][0].longitude==0 :
        min_dist = -1
        min_index = -1
        for j in list_classes[i]:
            if list_classes[j.dest_city][0].latitude==0 and list_classes[j.dest_city][0].longitude==0:
                continue
            else:
                if min_dist==-1:
                    min_dist=j.dist
                else:
                    if j.dist<min_dist:
                        min_dist=j.dist
                        min_index=list_classes[i].index(j)
        lat=list_classes[list_classes[i][min_index].dest_city][0].latitude
        lon=list_classes[list_classes[i][min_index].dest_city][0].longitude
        for k in list_classes[i]:
            k.latitude = lat
            k.longitude = lon


def is_goal(route, start_city, end_city):
    route_start= route.split()[0]
    route_end= route.split()[-1]
    if route_start == start_city and route_end == end_city :
        return True
    else:
        return False

def successors(city):
    return [i.dest_city for i in list_classes[city]]

def solve_bfs(start_city, end_city):
    fringe = [start_city]
    closed =[]
    while len(fringe) > 0:
        route_so_far = fringe.pop()
        city = route_so_far.split()[-1]
        for next_route in successors(city):
            new_route_so_far = route_so_far + " " + next_route
            k=new_route_so_far.split()[-1:-3:-1]
            k1=k[::-1]
            if is_goal(new_route_so_far, start_city, end_city):
                return(new_route_so_far)
            if k not in closed:
                fringe.insert(0, new_route_so_far)
                closed.append(k)

    return False


def solve_dfs(start_city, end_city):
    fringe = [start_city]
    closed = []
    while len(fringe) > 0:
        route_so_far = fringe.pop()

        city = route_so_far.split()[-1]
        for next_route in successors(city):
            new_route_so_far = route_so_far + " " + next_route

            k=new_route_so_far.split()[-1:-3:-1]
            k1=k[::-1]

            if is_goal(new_route_so_far, start_city, end_city):
                return(new_route_so_far)
            if k not in closed:
                fringe.append(new_route_so_far)
                closed.append(k)

    print('No solution found')
    return False


def solve_ids(start_city, end_city):
    depth = 2

    while True:
        fringe = [start_city]
        closed = []
        while len(fringe) > 0:

            route_so_far = fringe.pop()

            if len(route_so_far.split())-1 > depth :
                continue
            city = route_so_far.split()[-1]

            for next_route in successors(city):
                new_route_so_far = route_so_far + " " + next_route

                k=new_route_so_far.split()[-1:-3:-1]
                k1=k[::-1]

                if is_goal(new_route_so_far, start_city, end_city):
                    return(new_route_so_far)
                if k not in closed:
                    fringe.append(new_route_so_far)
                    closed.append(k)
        depth+=1

def solve_uniform(start_city, end_city, cost, list_classes):
    fringe = PriorityQueue()
    fringe.put([1,start_city])
    closed = []
    while not fringe.empty():
        route_so_far = fringe.get()

        city = route_so_far[1].split()[-1]


        for next_route in successors(city):
            new_route_so_far = route_so_far[1] + " " + next_route
            k=new_route_so_far.split()[-1:-3:-1]
            k1=k[::-1]

            dist=0
            time=0
            solution_list = new_route_so_far.split()
            for i in range(0,len(solution_list)-1):
                for next_city in list_classes[solution_list[i]]:
                    if str(next_city.dest_city ) == solution_list[i+1]:
                        dist+=next_city.dist
                        time+=float(next_city.dist)/float(next_city.speedlim)

            if is_goal(new_route_so_far, start_city, end_city):
                return(new_route_so_far)
            if k not in closed:
                if cost =='distance':
                    fringe.put((dist,new_route_so_far))
                if cost =='time':
                    fringe.put((time,new_route_so_far))
                if cost=='segments':
                    fringe.put((int(len(new_route_so_far)) , new_route_so_far))
                closed.append(k)
    print("No solution found")
    return False

def astar(start_city, end_city, cost):
    fringe = PriorityQueue()
    fringe.put((1,start_city))
    closed = []
    while not fringe.empty():
        route_so_far = fringe.get()
        city = route_so_far[1].split()[-1]

        for next_route in successors(city):

            new_route_so_far = route_so_far[1] + " " + next_route
            k=new_route_so_far.split()[-1:-3:-1]
            k1=k[::-1]

            dist=0
            time=0
            solution_list = new_route_so_far.split()
            for i in range(0,len(solution_list)-1):
                for next_city in list_classes[solution_list[i]]:
                    if str(next_city.dest_city ) == solution_list[i+1]:
                        dist+=next_city.dist
                        time+=float(next_city.dist)/float(next_city.speedlim)

            if is_goal(new_route_so_far, start_city, end_city):
                return(new_route_so_far)

            lon1,lat1 = list_classes[next_route][0].longitude, list_classes[next_route][0].latitude
            lon2,lat2 = list_classes[end_city][0].longitude, list_classes[end_city][0].latitude
            h = int(haversine(lon1, lat1, lon2, lat2))
            if k not in closed:
                if cost =='distance':
                    final = dist + h
                    fringe.put((final,new_route_so_far))
                if cost =='time':
                    final = time + h/avg_speedlim
                    fringe.put((final,new_route_so_far))
                if cost=='segments':
                    final=1
                    for nxt in list_classes[next_route]:
                        if nxt.name == end_city:
                            final=0
                    final+=len(new_route_so_far)
                    fringe.put((final , new_route_so_far))
                closed.append(k)
    print("No solution found")
    return False

def print_solution(solution_route):
    solution_list = solution_route.split()
# print(solution_route)
    dist = 0
    time = 0
    for i in range(0,len(solution_list)-1):
        for next_city in list_classes[solution_list[i]]:
            if str(next_city.dest_city ) == solution_list[i+1]:
                dist+=next_city.dist
                time+=float(next_city.dist)/float(next_city.speedlim)

    time = format(time, '.4f')
    return str(dist)+' '+str(time)+' '+solution_route

start_city=sys.argv[1]
end_city = sys.argv[2]

optimal = "no"
routing_algorithm = sys.argv[3]
cost_function = sys.argv[4]

if routing_algorithm=='bfs':
    if cost_function =='segments':
        optimal='yes'
    solution_route = solve_bfs(start_city,end_city)
    print(optimal+' '+print_solution(solution_route))
elif routing_algorithm=='uniform':
    optimal='yes'
    solution_route = solve_uniform(start_city,end_city,cost_function,list_classes)
    print(optimal+' '+print_solution(solution_route))
elif routing_algorithm=='dfs':
    solution_route = solve_dfs(start_city,end_city)
    print(optimal+' '+print_solution(solution_route))
elif routing_algorithm=='ids':
    if cost_function == 'segments':
        optimal = 'yes'
    solution_route=solve_ids(start_city,end_city)
    print(optimal+' '+print_solution(solution_route))
elif routing_algorithm=='astar':
    optimal='yes'
    solution_route = astar(start_city,end_city,cost_function)
    print(optimal+' '+print_solution(solution_route))
else:
    pass
