import os
import torch
import torchvision
import numpy as np
import pandas as pd
import plotly.express as px 
from torch.utils.data import DataLoader
from dataset_loader import load_np_dataset, get_subclass_dataset
from sklearn.manifold import TSNE


def tsne(model, args):

    np_test_img_path = os.path.join(args.config['generalization_path'], 'CIFAR10_Test_AC_6/rot90.npy')
    np_test_target_path = os.path.join(args.config['generalization_path'], 'CIFAR10_Test_AC_6/labels_test.npy')
    cifar10_path = args.config['data_path']
    test_data = torchvision.datasets.CIFAR10(
        cifar10_path, train=False, transform=torchvision.transforms.ToTensor(), download=True)

    test_data2 = load_np_dataset(np_test_img_path, np_test_target_path, torchvision.transforms.ToTensor(), dataset='cifar10', train=False)

    eval_in_data = get_subclass_dataset(test_data, args.one_class_idx)
    eval_out_data = get_subclass_dataset(test_data, 6)
    eval_out_data2 = get_subclass_dataset(test_data2, args.one_class_idx)
    eval_in = DataLoader(eval_in_data, shuffle=False, batch_size=16)
    eval_out = DataLoader(eval_out_data, shuffle=False, batch_size=16)
    eval_out2 = DataLoader(eval_out_data2, shuffle=False, batch_size=16)
    f_in = []
    f_out = []
    f_out2 = []
    model.eval()
    with torch.no_grad():
        for d1, d2, d3 in zip(eval_in, eval_out, eval_out2):
            im1, _ = d1
            im2, _ = d2
            im3, _ = d3
            im1, im2, im3 = im1.to(args.device), im2.to(args.device), im3.to(args.device)
            _, out1 = model(im1, True)
            _, out2 = model(im2, True)
            _, out3 = model(im3, True)
            f_in.append(out1[-1].detach().cpu().numpy())
            f_out.append(out2[-1].detach().cpu().numpy())
            f_out2.append(out3[-1].detach().cpu().numpy())
    
    f_in = np.concatenate(f_in, axis=0)
    f_out = np.concatenate(f_out, axis=0)
    y = [0 for i in range(len(f_in))]
    y.extend([1 for i in range(len(f_out))])
    x = np.concatenate([f_in, f_out])
    tsne_plot(x, y, 'frog', args)

    f_out2 = np.concatenate(f_out2, axis=0)
    y = [0 for i in range(len(f_in))]
    y.extend([1 for i in range(len(f_out2))])
    x = np.concatenate([f_in, f_out2])
    tsne_plot(x, y, 'rot90', args)


def tsne_plot(x, y, name, args):
    tsne = TSNE(n_components=2, verbose=1, random_state=123, learning_rate='auto', init='pca')
    z = tsne.fit_transform(x) 
    df = pd.DataFrame()
    df["y"] = y
    df["component_1"] = z[:,0]
    df["component_2"] = z[:,1]
    label_str = [str(yy) for yy in y]
    # for l in df.y.tolist():
    #     if l == 1:
    #         label_str.append(class1)
    #     else:
    #         label_str.append(class2)

    fig = px.scatter(data_frame=df, x="component_1", y="component_2", color=label_str,
            labels={
                        "component_1": "Component 1",
                        "component_2": "Component 2",
                    },
                title=f"CLIP encoding normal vs {name}",
                width=1024, height=1024)

    fig.update_layout(
        font=dict(
            size=22,  # Set the font size here
        )
    )
    # fig.show()
    fig.write_image(args.save_path + "{}_normal_vs_{}_tsne.png".format(args.e_holder, name))

