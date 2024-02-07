import torch.nn.functional as F


def channel_f(f1,f2,is_test=False,save_mat=False):
    """
    Function for implementing channel fusion strategy. 
    param f1: Extracted features of IR image.
    param f2: Extracted features of RGB image.
    param is_test: Boolean for stating if the test is in progress.
    param save_mat: For saving the matrix.
    
    """
    if is_test:
        fp1 = (((f1.mean(2)).mean(2)).unsqueeze(2)).unsqueeze(3)
        fp2 = (((f2.mean(2)).mean(2)).unsqueeze(2)).unsqueeze(3)
    else:
        fp1 = F.avg_pool2d(f1, f1.size(2))
        fp2 = F.avg_pool2d(f2, f2.size(2))
    mask1 = fp1 / (fp1 + fp2)
    mask2 = 1 - mask1
    if save_mat:
        import scipy.io as io
        mask = mask1.cpu().detach().numpy()
        io.savemat("./outputs/fea/mask.mat", {'mask': mask})
    return f1 * mask1 + f2 * mask2
