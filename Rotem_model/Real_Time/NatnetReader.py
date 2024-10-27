from os import listdir
from os.path import isfile, join
import os 
import yaml
import argparse
import pandas as pd

from data.NatNetClient import NatNetClient

class NatNetReader:

    def __init__(self):

        self.natnet = self.init_natnetClient()

        self.motive_matcher = {
                0: 'table_base',
                1: 'chest',
                2: 'shoulder',
                3: 'elbow',
                4: 'wrist',
                }


    def receiveRigidBodyList(self,rigidBodyList, timestamp):
        for (ac_id, pos, quat, valid) in rigidBodyList:
            if not valid:
                continue

    def init_natnetClient(self):
            
        # start natnet interface
        return NatNetClient(rigidBodyListListener=self.receiveRigidBodyList,server="132.66.51.232")#rigidBodyListListener=receiveRigidBodyList)


    def read_sample(self)-> dict:
        # xyz 
        locations = {
                'chest':[],
                'shoulder':[],
                'elbow':[],
                'wrist':[],
                'table_base':[],
                }
        #warmup
        for i in range(3):
            self.natnet.rigidBodyList

        rigid_bodys = self.natnet.rigidBodyList
        # print(rigid_bodys)

        for j in range(len(rigid_bodys)):
            locations[self.motive_matcher[rigid_bodys[j][0]]].append(rigid_bodys[j][1])

        return locations
