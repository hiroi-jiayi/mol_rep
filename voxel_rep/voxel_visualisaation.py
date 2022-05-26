# visualise using 3D voxel representation

def voxel_visual(grid_matrix):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    #norm= plt.Normalize(grid_matrix.min(), grid_matrix.max())
    ax.voxels(grid_matrix, edgecolor="k")
    plt.show()
    return 
 