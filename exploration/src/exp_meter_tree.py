#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 27 17:38:08 2019

@author: Toukir
"""
from graphviz import Digraph
from graphviz import Graph

from appai_lib import my_io
import math

all_nan = [59, 100, 158, 170, 224, 336, 465, 505, 588]
zero_sum =  [6, 7, 8, 9, 10, 16, 18, 23, 30, 36, 39, 41, 49, 55, 59, 70, 84, 97, 98, 100, 110, 112, 119, 131, 137, 158, 170, 180, 186, 192, 196, 208, 212, 213, 216, 224, 229, 230, 233, 235, 236, 237, 244, 246, 252, 256, 259, 260, 264, 266, 269, 270, 275, 279, 280, 282, 284, 288, 294, 298, 300, 309, 310, 315, 316, 323, 325, 328, 332, 334, 335, 336, 338, 340, 345, 351, 352, 356, 359, 364, 376, 382, 392, 398, 406, 410, 415, 420, 422, 423, 426, 429, 433, 437, 441, 442, 451, 455, 456, 457, 461, 462, 464, 465, 471, 477, 479, 480, 482, 486, 496, 497, 499, 505, 513, 523, 526, 535, 538, 544, 547, 555, 559, 573, 578, 585, 586, 588, 590, 596, 599]

dead = all_nan + zero_sum

def get_leaf_nodes(meter_tree,leaf_nodes,non_leaf_nodes):
    for key, value in meter_tree.items():
       if len(value.keys()) != 0:
           non_leaf_nodes.append(key)
           get_leaf_nodes(value,leaf_nodes,non_leaf_nodes)
       else:
           leaf_nodes.append(key)

def create_graph(meter_tree, dot, parent):
    for key, value in meter_tree.items():
        #if value !={}:

        row = meter_table.loc[meter_table.meter_id == key]
        if int(row.iloc[0].meter_type) == 2 :
            dot.attr("node", color='lightblue2',style='filled')
        else:
            dot.attr("node",color='crimson',style='filled')
        s = ""
        if key in dead:
            s = "\nDEAD"
        dot.node(str(key), str(key) + s)
        if parent != None:
            dot.edge(str(parent), str(key))
        if isinstance(value,dict):
            create_graph(value, dot, key)


def find(d,meter_id):
    found = None
    for key, value in d.items():
       if key == meter_id:
           return key
       if isinstance(value, dict):
           found = find(value,meter_id)
       if found:
           return found

def insert(d,meter_id, parent_id):
    found = False
    for key, value in d.items():
       if key == parent_id:
           if isinstance(value,dict):
               value[meter_id] = {}
           return True
       if isinstance(value, dict):
           found = insert(value,meter_id,parent_id)
       if found:
           return found


#my_io.get_meter_table(my_io.mydb,"../data/raw/ems_meter")
meter_table = my_io.load_data("../data/raw/ems_meter.csv")

meter_tree ={}


isMissing = True
while isMissing:
    for index,row in meter_table.iterrows():
        isMissing = False
        x = find(meter_tree,row.meter_id)
        if x == None:
            isMissing = True
            if math.isnan(row.parent_id):
                meter_tree[row.meter_id] = {}

            else:
                parent = find(meter_tree, row.parent_id)
                if parent != None:
                    insert(meter_tree, row.meter_id, row.parent_id)

dot = Digraph(comment="The meter tree",engine='neato')
dot.attr(width="2800pt", height="2800pt",fixedsize='true')
dot.graph_attr.update(overlap="false")
create_graph(meter_tree,dot,None)

leaf_nodes = []
non_leaf_nodes = []
get_leaf_nodes(meter_tree,leaf_nodes,non_leaf_nodes)

real_meters = meter_table.loc[meter_table.meter_type== 2].meter_id
virtual_meters = meter_table.loc[meter_table.meter_type != 2].meter_id


#print(dot.source)
#print(meter_tree)
#dot.render("Meter relations tree", view =  True)