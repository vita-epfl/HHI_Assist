import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import os
from PIL import Image
import io



skeleton = [[8,0],[0,1],[1,2],[2,3],[8,4],[4,5],[5,6],[6,7],[8,9],[18,10],[10,11],[11,12],
                [12,13],[18,14],[14,15],[15,16],[16,17],[9,18],[18,19]]

color_pairs = colors = [
    ((100/255, 140/255, 140/255), (40/255, 80/255, 80/255)),   # Dark Teal
    ((130/255, 180/255, 210/255), (70/255, 120/255, 150/255)),   # Blueish Tone
    ((210/255, 160/255, 180/255), (150/255, 100/255, 120/255))  # Pink 
]

def _setup_axes(ax):
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    
    #rotate the axes
    ax.view_init(elev=80, azim=-90) #120 -60 , 80 -90
    
    ax.set_box_aspect([(max_all_xyz[0]-min_all_xyz[0])/(max_all_xyz[1]-min_all_xyz[1]),
                       1,
                       (max_all_xyz[2]-min_all_xyz[2])/(max_all_xyz[1]-min_all_xyz[1])])
    
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    
    ax.grid(False)
    ax.set_axis_off()
        
    ax.axes.set_xlim3d(left=min_all_xyz[0], right=max_all_xyz[0]) 
    ax.axes.set_ylim3d(bottom=min_all_xyz[1], top=max_all_xyz[1]) 
    ax.axes.set_zlim3d(bottom=min_all_xyz[2], top=max_all_xyz[2]) 
        
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    ax.margins(0) #remove margins
    

def _plot_skeleton(ax, data, label, title, seq_number=1):
    xdata, ydata, zdata = data.T
    ax.scatter(xdata, ydata, zdata, color="black", s=0.5) #"turquoise" label=label, 

    n_joints = xdata.shape[0]
    
    l_j = 10 if seq_number==1 else 13
    for j in range(n_joints-1):
        color_indx = seq_number % len(color_pairs) - 1
        
        #COLOR:
        if seq_number>2:
            color_indx = 2
        
        selected_pair = color_pairs[color_indx]
        color =  selected_pair[0] if j < l_j else selected_pair[1] #("lightseagreen" if j<l_j else "turquoise") if seq_number%2==1 else ("palevioletred" if j<l_j else "pink")
        ax.plot(xdata[skeleton[j]], ydata[skeleton[j]], zdata[skeleton[j]] , color =color)
        
    _setup_axes(ax)
    
def visualize_sequence(sequences, save_path, string="test_vis_seq", return_array=False, project_under=False, title=None):
        
    if not sequences:
        raise ValueError("No sequences provided")
    
    # Determine the number of columns based on the first sequence's joint count
    n_columns = 6 #9 if sequences[0].shape[0] == 25 else 8
    total_frames = sum(seq.shape[0] for seq in sequences)
    
    # Calculate skip rates for each sequence to limit them to a single row per sequence if necessary
    skip = max([seq.shape[0] for seq in sequences]) // n_columns #+ 1
    
    # Subsample sequences based on calculated skip
    sequences_subsampled = [seq[::skip,:,:] for seq in sequences]
    
    #reducing the 0 joint from all the other joint in all the skeletons in the sequences:
    sequences_subsampled = [seq[:,1:,:]-seq[:,:1,:] for seq in sequences_subsampled]
    #making the fist joint zero:
    sequences_subsampled = [np.concatenate((np.zeros((seq.shape[0],1,3)),seq), axis=1) for seq in sequences_subsampled]
    
    # Determine the total number of rows needed24
    n_rows = sum((seq.shape[0] - 1) // n_columns + 1 for seq in sequences_subsampled)
    fig = plt.figure(figsize=(n_columns, n_rows))
    
    all_xdata = np.concatenate([seq[:, :, 0].flatten() for seq in sequences_subsampled])
    all_ydata = np.concatenate([seq[:, :, 1].flatten() for seq in sequences_subsampled])
    all_zdata = np.concatenate([seq[:, :, 2].flatten() for seq in sequences_subsampled])

    global max_all_xyz , min_all_xyz
    max_all_xyz = [np.max(all_xdata), np.max(all_ydata), np.max(all_zdata)]
    min_all_xyz = [np.min(all_xdata), np.min(all_ydata), np.min(all_zdata)]
    
    current_frame = 0
    for seq_idx, seq in enumerate(sequences_subsampled):
        for frame_idx in range(seq.shape[0]):
            ax = fig.add_subplot(n_rows, n_columns, current_frame + 1, projection='3d')
            
            if project_under:
                under_seq = sequences_subsampled[1]
                _plot_skeleton(ax, under_seq[frame_idx, :, :], label=f"pose|{current_frame}", title=f"frame {current_frame}", seq_number= 1)

            _plot_skeleton(ax, seq[frame_idx, :, :], label=f"pose|{current_frame}", title=f"frame {current_frame}", seq_number=seq_idx + 1)
            
            current_frame += 1
        # Adjust for any empty spaces in the last row of each sequence
        while current_frame % n_columns != 0:
            fig.add_subplot(n_rows, n_columns, current_frame + 1)  # Add empty subplots
            plt.axis('off')
            current_frame += 1

    plt.subplots_adjust(wspace=0, hspace=0)
    # plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)s
    
    if return_array:
        fig_aux = plt.figure(figsize=(1, n_rows))
        for i in range(len(sequences)):
            ax = fig_aux.add_subplot(n_rows, 1, i + 1, projection='3d')
            for k in range(sequences[i].shape[0]):
                _plot_skeleton(ax, sequences[i][k, 0], label="", title="", seq_number=i + 1)
                
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')#, pad_inches=1)
        buf.seek(0)
        image = Image.open(buf)
        image_array = np.array(image)
        buf.close()
        
        buf_aux = io.BytesIO()
        fig_aux.savefig(buf_aux, format='png', bbox_inches='tight', pad_inches=0)
        buf_aux.seek(0)
        image_aux = Image.open(buf_aux)
        image_aux_array = np.array(image_aux)
        buf_aux.close()
        image_array = np.concatenate((image_array, image_aux_array), axis=1)
        plt.close()
        return image_array
    
    else:
        # Split the string into parts
        parts = string.split('/')
        
        # Create the directory path
        directory = os.path.join(save_path, *parts[:-1]) #, "plots"
        
        # Create the directories
        os.makedirs(directory, exist_ok=True)
        
        if title is not None:
            assert type(title) == str, "Title must be a string"

            plt.title(title)
            
        # Save the figure
        plt.savefig(os.path.join(directory, "{}.png".format(parts[-1])), bbox_inches='tight', pad_inches=0)
        plt.savefig(os.path.join(directory, "{}.pdf".format(parts[-1])), bbox_inches='tight', pad_inches=0)
        plt.savefig(os.path.join(directory, "{}.svg".format(parts[-1])), bbox_inches='tight', pad_inches=0)
        # print("Saved!", os.path.join(directory, "{}.png".format(parts[-1])))
        plt.close()


def vis_link_len(sequences, save_path, string="test_link_len"):
    #T,J,3    
    #subplot, 4*5
        
    def get_or_create_subplot(fig, nrows, ncols, index):
        # Calculate the position of the subplot
        row = (index - 1) // ncols
        col = (index - 1) % ncols
        
        # Check if the subplot already exists
        for ax0 in fig.get_axes():
            if ax0.get_subplotspec().rowspan.start == row and ax0.get_subplotspec().colspan.start == col:
                return ax0
        # If it doesn't exist, create a new one
        return fig.add_subplot(nrows, ncols, index)
        
    fig = plt.figure(figsize=(20, 15))
    # print(sequences.shape)
    
    times = np.linspace(-23, 24, num=48, dtype=int)

    if len(sequences.shape) == 4:
        n_s = sequences.shape[0]
    else:
        n_s = 1
        
    for j in range(n_s):
        if n_s>1:
            sequence = sequences[j]
        else:
            sequence = sequences
        for i in range(len(skeleton)):
            joint_1 = sequence[:,skeleton[i][0],:]
            joint_2 = sequence[:,skeleton[i][1],:]

            link_len = np.linalg.norm(joint_1-joint_2, axis=-1)
            ax = get_or_create_subplot(fig, 4, 5, i + 1) #ax = fig.add_subplot(4, 5, i + 1)

            l = link_len[0]
            link_len = link_len - l #to plot the length difference from the first frame
            
            # if np.abs(link_len.max()) > 0.18: #to visualize the sequence with a link which is changing more than the threshold 
            #     visualize_sequence([sequence], save_path, string=f"test_{i,j}", return_array=False, project_under=False, title=None)
            
            ax.plot(times, link_len*1000, linewidth=2, color = "steelblue", alpha=0.2)
            ax.set_title(f"link {i}")
            #add labels to axis:
            ax.set_xlabel('time step', fontsize=18)
            ax.set_ylabel('link length [mm]', fontsize=16)
            ax.tick_params(axis='both', which='major', labelsize=16)
            ax.tick_params(axis='both', which='minor', labelsize=14)
            
            ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
            
            x_ticks = np.linspace(-23, 24, num=5, dtype=int)
            # y_ticks = np.linspace(0, int(np.max(link_len * 1000)), num=5, dtype=int)
            ax.set_xticks(x_ticks)
            # ax.set_yticks(y_ticks)
        
    # plt.subplots_adjust(wspace=0, hspace=0)
    plt.tight_layout()
    parts = string.split('/')
    directory = os.path.join(save_path, *parts[:-1])
    os.makedirs(directory, exist_ok=True)
    plt.savefig(os.path.join(directory, "{}_.png".format(parts[-1])), pad_inches=0)
    plt.savefig(os.path.join(directory, "{}_.pdf".format(parts[-1])), pad_inches=0)
    plt.close()
    # breakpoint()
  
    
def cal_abs_diff(x, y_preds):
    
    #x: B,J,3
    #y_preds: B,T,J,3
            
    diff_sum = np.empty((len(skeleton), y_preds.shape[1]))
            
    for i in range(len(skeleton)):
        joint_1 = x[:,skeleton[i][0],:]
        joint_2 = x[:,skeleton[i][1],:]
        link_len_ref = np.linalg.norm(joint_1-joint_2, axis=-1)
        link_len_ref = np.expand_dims(link_len_ref, axis=1)
        
        # for t in range(y_preds.shape[1]):
        joint_1 = y_preds[:,:,skeleton[i][0],:]
        joint_2 = y_preds[:,:,skeleton[i][1],:]
        link_len = np.linalg.norm(joint_1-joint_2, axis=-1)
    
        diff = np.abs(link_len_ref-link_len)
        
        diff_sum[i] = np.sum(diff, axis=0)
             
    return diff_sum