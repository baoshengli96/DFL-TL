from tools import Initialize
from clients import clientDFLProto
import time
import copy
import numpy as np
from optimize_proto_diff import ProtoTopologyOptimizer


class DFLTL(Initialize):
    def __init__(self, args, times):
        super().__init__(args, times)
        # generate the set of clients
        self.clientSets(args, clientDFLProto)
        self.Budget = []
        self.all_protos = {}
        self.num_classes = args.num_classes
        self.zeta = args.zeta if hasattr(args, 'zeta') else 0.01  
        self.min_diag_value = args.min_diag_value if hasattr(args, 'min_diag_value') else 0.1        
        self.get_modelsize()

        self.initialize_devices()       

    def train(self):
        topology_opt = ProtoTopologyOptimizer(zeta=self.zeta, min_diag_value=self.min_diag_value, max_iter=100)
        
        for round_num in range(self.global_rounds):
            s_t = time.time()         
            if round_num % self.eval_gap == 0:  
                print(f"\n-------------Round number: {round_num}-------------")
                print("\nEvaluate global model")
                self.evaluate()


            self.build_topology()


            for client in self.clients:
                client.train()

            self.get_all_protos()

            weight_matrix, slem_opt, slem_mh = topology_opt.optimize(all_protos=self.all_protos, adjacency_matrix=self.adjacency_matrix, num_classes=self.num_classes)                
            self.slem_opts[round_num] = slem_opt  # the slem got from topology_opt
            self.slem_mh[round_num] = slem_mh     # the slem of M-H method       
            print('slem_opt: ',slem_opt)
            print('slem_mh: ', slem_mh)


            self.communicate(weight_matrix)
            

            local_epoch = 1  
            round_compute_latencies = [d.compute_latency(self.d_C, local_epoch) for d in self.devices]
            round_compute_latency = sum(round_compute_latencies)
            round_compute_energy = sum([d.compute_energy(self.d_C, self.kappa, local_epoch) for d in self.devices])
            

            # print(f"DEBUG - Compute latency: {round_compute_latency:.6f}s")
            # print(f"DEBUG - Compute energy: {round_compute_energy:.6f}J")
            

            device_comm_latencies = []
            round_comm_energy = 0
            
            for i in range(len(self.devices)):
                device_max_latency = 0
                device_energy = 0
                
                for j in range(len(self.devices)):
                    if self.adjacency_matrix[i][j] == 1:
    
                        latency = self.calculate_transmission_latency(i, j)
                        if latency != float('inf'):
                            device_max_latency = max(device_max_latency, latency)
                        
           
                        energy = self.calculate_transmission_energy(i, j)
                        device_energy += energy
                
                if device_max_latency > 0:
                    device_comm_latencies.append(device_max_latency)
                round_comm_energy += device_energy
            

            # if device_comm_latencies:
            #     print(f"DEBUG - Max comm latency: {max(device_comm_latencies):.6f}s")
            # print(f"DEBUG - Total comm energy: {round_comm_energy:.6f}J")
  
            round_comm_latency = sum(device_comm_latencies)
            total_latency = round_compute_latency + round_comm_latency
            total_energy = round_compute_energy + round_comm_energy
        
            

            self.latencies.append(total_latency)
            self.energies.append(total_energy)
            
            self.Budget.append(time.time() - s_t)
            print('-'*50, self.Budget[-1])
            print(f'Round Latency: {total_latency:.4f}s, Energy: {total_energy:.4f}J')

        print("\nAverage time per round.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))
        print(f"\nAverage latency per round: {sum(self.latencies)/len(self.latencies):.4f}s")
        print(f"Average energy per round: {sum(self.energies)/len(self.energies):.4f}J")

        self.save_results(self.Budget)

    def update_models(self):
        for client in self.clients:
            client.set_parameters(self.old_models[client.id])

    def communicate(self, weight_matrix):
        self.old_models = []
        for client in self.clients:
            i = client.id
            old_model = copy.deepcopy(client.model)
            for param in old_model.parameters():
                param.data.zero_()
            # communicate
            for client_j in self.clients:
                for w_new, w_j in zip(old_model.parameters(), client_j.model.parameters()):
                    w_new.data += w_j.data.clone() * weight_matrix[i,client_j.id]                
            self.old_models.append(old_model)
        self.update_models()

    def get_all_protos(self):
        has_protos = all(hasattr(client, 'protos') for client in self.clients)
        if has_protos:
            for i, client in enumerate(self.clients):
                if hasattr(client, 'protos') and client.protos:
                    self.all_protos[i] = client.protos        
   