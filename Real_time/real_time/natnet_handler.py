import logging
from natnet.NatNetClient import NatNetClient

logger = logging.getLogger(__name__)

class NatNetReader:
    def __init__(self, config):
        self.natnet = self.init_natnetClient(config)
        self.motive_matcher = {
            0: 'table_base',
            1: 'chest',
            2: 'shoulder',
            3: 'elbow',
            4: 'wrist',
        }
        self.connected = False

    def receiveRigidBodyList(self, rigidBodyList, timestamp):
        self.rigidBodyList = rigidBodyList
        self.timestamp = timestamp
        for (ac_id, pos, quat, valid) in rigidBodyList:
            if not valid:
                continue

    def init_natnetClient(self, config):
        # start natnet interface
        return NatNetClient(
            rigidBodyListListener=self.receiveRigidBodyList,
            server=config['natnet_server_ip'],
            multicast=config['natnet_multicast_ip'],
            commandPort=config['natnet_command_port'],
            dataPort=config['natnet_data_port']
        )
    def read_sample(self) -> dict:
        # xyz
        locations = {
            'chest': [],
            'shoulder': [],
            'elbow': [],
            'wrist': [],
            'table_base': [],
        }
        if not self.connected:
            logger.warning("NatNet is not connected, returning default values for ground truth.")
            return locations
        # Get the latest data from rigid body list
        rigid_bodys = self.rigidBodyList
        for j in range(len(rigid_bodys)):
            locations[self.motive_matcher[rigid_bodys[j][0]]].append(rigid_bodys[j][1])

        return locations

    def connect(self):
        try:
            self.natnet.run()
            self.connected = True
        except Exception as e:
            logger.error("Failed to connect to NatNet server: %s", e)
            self.connected = False

    def disconnect(self):
        self.natnet.stop()
        self.connected = False