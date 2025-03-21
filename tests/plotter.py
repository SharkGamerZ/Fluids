import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
from matplotlib.widgets import Button


# Generate plots for the performance of the algorithms
def generate_plots(in_csv, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(in_csv)

    has_cuda = 'Gauss_CUDA' in df.columns and 'Jacobi_CUDA' in df.columns

    # Group by Matrix_Size first, then plot each size separately
    for matrix_size in df['Matrix_Size'].unique():
        # Filter data for this matrix size
        df_size = df[df['Matrix_Size'] == matrix_size]

        plt.figure(figsize=(10, 6))

        plt.plot(df_size['Iterations'], df_size['Gauss_Serial'], label='Gauss Serial', marker='o')
        plt.plot(df_size['Iterations'], df_size['Gauss_OMP'], label='Gauss OMP', marker='s')
        if has_cuda:
            plt.plot(df_size['Iterations'], df_size['Gauss_CUDA'], label='Gauss CUDA', marker='d')
        plt.plot(df_size['Iterations'], df_size['Jacobi_Serial'], label='Jacobi Serial', marker='^')
        plt.plot(df_size['Iterations'], df_size['Jacobi_OMP'], label='Jacobi OMP', marker='x')
        if has_cuda:
            plt.plot(df_size['Iterations'], df_size['Jacobi_CUDA'], label='Jacobi CUDA', marker='*')

        plt.xlabel('Iterations')
        plt.ylabel('Time (microseconds)')
        plt.title(f'Algorithm Performance - Matrix Size {matrix_size}')

        plt.annotate('Lower is better ↓',
                     xy=(0.02, -0.10), xycoords='axes fraction',
                     fontsize=10, color='darkred', fontweight='bold',
                     bbox=dict(boxstyle='round,pad=0.3', fc='lightyellow', ec='orange', alpha=0.8))

        plt.legend()
        plt.grid(True)

        plt.minorticks_on()
        plt.grid(which='minor', linestyle=':', alpha=0.4)

        plt.savefig(f'{output_dir}/plot_size_{matrix_size}.png', dpi=150)
        plt.close()

    print(f'Generated {len(df["Matrix_Size"].unique())} plots in the "{output_dir}" directory')


# Viewer code to navigate through plots
def view_plots(output_dir):
    plots = sorted(glob.glob(f'{output_dir}/plot_size_*.png'),
                   key=lambda plot: int(os.path.basename(plot).replace('plot_size_', '').replace('.png', '')))
    if not plots:
        print("No plots found!")
        return

    print("Viewing plots - use the Previous/Next buttons to navigate")
    current_idx = [0]

    fig, ax = plt.subplots(figsize=(12, 8))
    plt.subplots_adjust(bottom=0.2)

    def show_plot(idx):
        ax.clear()
        img = plt.imread(plots[idx])
        ax.imshow(img)
        ax.axis('off')
        matrix_size = os.path.basename(plots[idx]).replace('plot_size_', '').replace('.png', '')
        ax.set_title(f'Plot {idx + 1}/{len(plots)}: Matrix Size {matrix_size}', fontsize=14)
        fig.canvas.draw_idle()

    def next_plot(event):
        current_idx[0] = (current_idx[0] + 1) % len(plots)
        show_plot(current_idx[0])

    def prev_plot(event):
        current_idx[0] = (current_idx[0] - 1) % len(plots)
        show_plot(current_idx[0])

    ax_prev = plt.axes((0.4, 0.05, 0.1, 0.075))
    ax_next = plt.axes((0.55, 0.05, 0.1, 0.075))
    btn_prev = Button(ax_prev, 'Previous')
    btn_next = Button(ax_next, 'Next')
    btn_prev.on_clicked(prev_plot)
    btn_next.on_clicked(next_plot)

    show_plot(0)
    plt.show()


if __name__ == "__main__":
    # CSV at ../cmake-build-release/results.csv when generated by the CMake build
    generate_plots('results.csv', 'matrix_plots')
    view_plots('matrix_plots')
