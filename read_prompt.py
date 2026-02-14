import torch

# 假设你的.pth文件名为 'model.pth'
file_path = 'checkpoints/9_12_4_multiscale_visa/epoch_15.pth'

# 读取.pth文件
loaded_object = torch.load(file_path)

#print(loaded_object.shape)
# 如果文件包含的是模型参数（state_dict），你可以这样加载它们到模型中
# 假设你已经有一个定义好的模型类 MyModel
# model = MyModel()
# model.load_state_dict(loaded_object)
# model.eval()  # 设置模型为评估模式

print(loaded_object['prompt_learner'].keys())  # 打印加载的对象以查看其内容
print('ctx_pos.shape,ctx_neg.shape',loaded_object['prompt_learner']['ctx_pos'].shape,loaded_object['prompt_learner']['ctx_neg'].shape)
# ctx_pos.shape,ctx_neg.shape torch.Size([1, 1, 12, 768]) torch.Size([1, 1, 12, 768])