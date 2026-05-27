import socket
import threading
import numpy as np
import time

import rclpy
from rclpy.node import Node
from loihi_power_msgs.msg import Power

class PowerPublisher(Node):
    def __init__(self):
        super().__init__('loihi_power')
        self.publisher_ = self.create_publisher(Power, 'loihi_power', 1)
        self.dynamic_power_publisher_ = self.create_publisher(Power, 'loihi_dynamic_power', 1)
        self.idle_power_publisher_ = self.create_publisher(Power, 'loihi_idle_power', 1)
        
        self.declare_parameter('NXSDKHOST', '10.42.0.100')
        self.board = self.get_parameter('NXSDKHOST').get_parameter_value().string_value
        self.get_logger().info(f'Using board: {self.board}')
        self.declare_parameter('pac_socket', 7225)
        self.pac_port = self.get_parameter('pac_socket').get_parameter_value().integer_value
        self.get_logger().info(f'Using PAC port: {self.pac_port}')
        self.declare_parameter('moving_average_window', 100)
        self.moving_average_window = int(self.get_parameter('moving_average_window').value)
        self.get_logger().info(f'Moving average window is set to: {self.moving_average_window}')
        
        self.keep_socket_open = True
        self._raw_data = None
        self.get_logger().info(
            f'Expecting an already-running PAC socket collector on {self.board}:{self.pac_port}'
        )
        
        self.idle_power = None
        if self.moving_average_window > 0:
            self.power_list = {
                'vddm': [],
                'vddio': [],
                'vdd': []
            }
          
        
    def compute_idle_power(self, measure_time=5.0):
        while self._raw_data is None:
            time.sleep(0.01)
        time_start = time.time()
        idle_power_data = []
        while time.time() - time_start < measure_time:
            idle_power_data.append(self._raw_data)
            time.sleep(0.01)
        self.idle_power = {
            'vddm': np.mean([data['vddm'] for data in idle_power_data]),
            'vddio': np.mean([data['vddio'] for data in idle_power_data]),
            'vdd': np.mean([data['vdd'] for data in idle_power_data]),
            'na': np.mean([data['na'] for data in idle_power_data]),
            'total': np.mean([data['total'] for data in idle_power_data])
        }
        
        # print(f'Idle power computed: {self.idle_power}')
        self.get_logger().info(f'Idle power computed: {self.idle_power}')
        
        
    def kill_power_daemon(self):
        self.get_logger().info('Leaving manually-started power collector running on the board.')
        
    def read_socket(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_TCP, socket.TCP_NODELAY, 1)
        s.settimeout(1.0)  # Set a timeout of 1 second
        deadline = time.monotonic() + 30.0
        while True:
            try:
                s.connect((self.board, self.pac_port))
                break
            except (socket.error, socket.timeout) as e:
                if time.monotonic() >= deadline:
                    self.get_logger().error(f"Failed to connect to {self.board} on port {self.pac_port}: {e}")
                    raise RuntimeError(f"Socket connection failed for board {self.board}") from e
                time.sleep(0.25)
        
        try:
            DATA_SIZE = 7 * 4   # 5 data points each of size float32
            while self.keep_socket_open:
                try:
                    data_rec =  np.frombuffer(s.recv(DATA_SIZE, socket.MSG_WAITALL), dtype=np.float32)
                    if len(data_rec) != 7:
                        self.get_logger().error(f"Received data of unexpected size: {len(data_rec)}")
                        continue
                    
                    self._raw_data = {
                        'id': int(data_rec[0]),
                        'time':  int(data_rec[1]),
                        'acc_count':  int(data_rec[2]),
                        'vddm':  float(data_rec[3]),
                        'vddio':   float(data_rec[4]),
                        'vdd':    float(data_rec[5]),
                        'na':   float(data_rec[6]),
                        'total':  float(data_rec[3]) + float(data_rec[4]) + float(data_rec[5])
                    }
                    
                    if self.idle_power is not None:
                        self.publish_power()             
                
                except socket.timeout:
                    continue
        
        except Exception as e:
            self.get_logger().error(f"Error reading from socket: {e}")
        finally:
            s.close()
    
    def publish_power(self):
        msg = Power()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'loihi_power_frame'
        if self._raw_data is None:
            self.get_logger().warn('No data received yet, skipping publish.')
            return
        msg.id = self._raw_data['id']
        msg.time = self._raw_data['time']
        msg.acc_count = self._raw_data['acc_count']
        if self.publisher_.get_subscription_count() > 0:
            msg.vddm = self._raw_data['vddm']
            msg.vddio = self._raw_data['vddio']
            msg.vdd = self._raw_data['vdd']
            msg.na = self._raw_data['na']
            msg.total = self._raw_data['total']
            self.publisher_.publish(msg)
        
        if self.dynamic_power_publisher_.get_subscription_count() > 0:
            if self.moving_average_window > 0:
                self.power_list['vddm'].append(self._raw_data['vddm'])
                self.power_list['vddio'].append(self._raw_data['vddio'])
                self.power_list['vdd'].append(self._raw_data['vdd'])
                while len(self.power_list['vddm']) > self.moving_average_window:
                    self.power_list['vddm'].pop(0)
                    self.power_list['vddio'].pop(0)
                    self.power_list['vdd'].pop(0)
                
                
                
                vddm = np.mean(self.power_list['vddm']) - self.idle_power['vddm']
                msg.vddm = vddm if vddm > 0 else 0.0
                vddio = np.mean(self.power_list['vddio']) - self.idle_power['vddio']
                msg.vddio = vddio if vddio > 0 else 0.0
                vdd = np.mean(self.power_list['vdd']) - self.idle_power['vdd']
                msg.vdd = vdd if vdd > 0 else 0.0
                # na = np.mean(self.power_list['na']) - self.idle_power['na']
                # msg.na = na if na > 0 else 0.0
                msg.na = 0.0
                msg.total = vdd + vddio + vddm
            else:
            
                vddm = self._raw_data['vddm'] - self.idle_power['vddm']
                msg.vddm =  vddm if vddm > 0 else 0.0
                vddio = self._raw_data['vddio'] - self.idle_power['vddio']
                msg.vddio = vddio if vddio > 0 else 0.0
                vdd = self._raw_data['vdd'] - self.idle_power['vdd']
                msg.vdd = vdd if vdd > 0 else 0.0
                na = self._raw_data['na'] - self.idle_power['na']
                msg.na = na if na > 0 else 0.0
                msg.total = vdd + vddio + vddm
            self.dynamic_power_publisher_.publish(msg)
        
        if self.idle_power_publisher_.get_subscription_count() > 0:
            msg.vddm = self.idle_power['vddm']
            msg.vddio = self.idle_power['vddio']
            msg.vdd = self.idle_power['vdd']
            msg.na = self.idle_power['na']
            msg.total = self.idle_power['total']
            self.idle_power_publisher_.publish(msg)
        
        # self.get_logger().info(f'Publishing: {msg.data}')

def main(args=None):
    rclpy.init(args=args)
    node = PowerPublisher()
    try:
        # Start the socket reading in a separate thread
        socket_thread = threading.Thread(target=node.read_socket)
        socket_thread.start()
        node.compute_idle_power()
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down...')
    finally:
        node.keep_socket_open = False
        socket_thread.join()
    node.kill_power_daemon()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
