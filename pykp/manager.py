import torch
import torch.nn as nn

class ManagerBasic(nn.Module):
    def __init__(self, goal_vector_size):
        super(ManagerBasic, self).__init__()
        self.goal_vector_size = goal_vector_size

        present_goal_vector = torch.zeros(self.goal_vector_size)
        absent_goal_vector = torch.zeros(self.goal_vector_size)
        # init uniformly
        initrange = 0.1
        present_goal_vector.uniform_(-initrange, initrange)
        absent_goal_vector.uniform_(-initrange, initrange)
        # set them to module parameters
        self.present_goal_vector = nn.Parameter(present_goal_vector, True)
        self.absent_goal_vector = nn.Parameter(absent_goal_vector, True)

    def forward(self, is_absent):
        """
        :param is_absent: tensor with size [batch_size]
        :return:
        """
        batch_size = is_absent.size()[0]
        g_t = []

        for i in range(batch_size):
            if int(is_absent[i].item()) == 1:
                g_t.append(self.absent_goal_vector)
            else:
                g_t.append(self.present_goal_vector)

        g_t = torch.stack(g_t, dim=0).unsqueeze(0)  # [1, batch_size, goal_vector_size]

        return g_t
