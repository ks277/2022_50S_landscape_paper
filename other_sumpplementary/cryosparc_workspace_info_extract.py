#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 13:59:15 2022

Run in a 
@author: shengkai
"""
import json
import numpy as np
import os
import pandas as pd
import sys

w_place = sys.argv[1]

log = open("info_%s_extraction.log"%w_place, "w") 

job_list = [i for i in os.listdir(".") if i[0] == "J"]
job_type_dic = {}
job_par_dic = {}
for i in job_list:
    
    try:
        temp_file = open("%s/job.json"%i)
        temp_json = json.load(temp_file)
        temp_file.close()
        if temp_json["workspace_uids"][0] == w_place:
            job_type_dic[i] = [temp_json["job_type"], temp_json["parents"], temp_json["children"]]
            if temp_json["job_type"] == "homo_abinit":
                job_dir_list = [j for j in os.listdir(i + "/")
                                if j.split(".")[-1]=="csg"]
                job_dir_list = [j for j in job_dir_list
                                if j.split("_")[-3]=="particles"]
                
                if len(job_dir_list) == 1:
                    temp_csg = open(i + "/" + job_dir_list[0])
                    temp_csg_l = list(temp_csg)
                    temp_csg.close()
                    for line in temp_csg_l:
                        if line.strip().split(":")[0] == "name": col = line.strip().split(":")[1] 
                        if line.strip().split(":")[0] == "num_items":
                            n_items_temp = int(line.strip().split(":")[1])
                            job_par_dic[i] = {col: n_items_temp}
                            break
                    continue
                temp_dic = {}
                for csg in job_dir_list:
                    temp_csg = open(i + "/" + csg)
                    temp_csg_l = list(temp_csg)
                    temp_csg.close()
                
                    for line in temp_csg_l:
                        if line.strip().split(":")[0] == "name": col = line.strip().split(":")[1].strip()
                        if line.strip().split(":")[0] == "num_items":
                            n_items_temp = int(line.strip().split(":")[1])
                            temp_dic[col] = n_items_temp
                            break
                    job_par_dic[i] = temp_dic
                        
    except:
        print("%s failed"%i)
        log.write("Extraction of %s failed\n"%i)
        continue
    

jid =  list(job_type_dic.keys())
jid.sort()

x = open("Info_%s.csv"%w_place, "w")
x.write("jid, job_type, parents, children, particle_num, particle_num_c1, particle_num_c2, particle_num_c3, particle_num_c4, particle_num_c5, \n")

for j in jid:
    s = ""
    s += j + ", " #jid
    s += job_type_dic[j][0] + ", " #job_type
    s += " ".join(job_type_dic[j][1]) + ", "  #parents
    s += " ".join(job_type_dic[j][2]) + ", "  #children
    try:
        temp_dic = job_par_dic[j]
        temp_key = list(temp_dic.keys())
        temp_key.sort()
        for p in temp_key:
            s += str(temp_dic[p]) + ", "
    except:
        print("Incomplete info for %s"%j)
        log.write("Incomplete info for %s\n"%j)
        continue
    s += "\n"
    x.write(s)

x.close()
log.close()
    