import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import gaussian_filter
import matplotlib.colors as mcolors

def moving_average(data, window_size=5):
    """Compute a moving average with reflection at the edges."""
    return data.rolling(window=window_size, min_periods=1, center=True).mean()


# 2D Plotting
def generate_plots(in_csv, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(in_csv)

    has_cuda = 'Gauss_CUDA' in df.columns and 'Jacobi_CUDA' in df.columns

    for matrix_size in df['Matrix_Size'].unique():
        df_size = df[df['Matrix_Size'] == matrix_size].copy()
        df_size.sort_values('Iterations', inplace=True)  # Ensure x-axis order

        # Apply moving average to each column
        for col in ['Gauss_Serial', 'Gauss_OMP', 'Jacobi_Serial', 'Jacobi_OMP']:
            df_size[col] = moving_average(df_size[col])
        if has_cuda:
            df_size['Gauss_CUDA'] = moving_average(df_size['Gauss_CUDA'])
            df_size['Jacobi_CUDA'] = moving_average(df_size['Jacobi_CUDA'])

        plt.figure(figsize=(10, 6))

        # Plot with moving average
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

        plt.yscale('log')
        plt.ylim(1, 500_000)

        plt.legend()
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)



        # Get Path
        matrix_padding = matrix_size.zfill(4)
        path = f'{output_dir}/plot_size_{matrix_padding}.png'

        plt.savefig(path, dpi=150)
        plt.close()

    print(f'Generated {len(df["Matrix_Size"].unique())} plots in the "{output_dir}" directory')



# 3d plots
def generate_3d_plot(in_csv, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(in_csv)
    has_cuda = 'Gauss_CUDA' in df.columns and 'Jacobi_CUDA' in df.columns

    methods = ['Gauss_Serial', 'Gauss_OMP', 'Jacobi_Serial', 'Jacobi_OMP']
    if has_cuda:
        methods += ['Gauss_CUDA', 'Jacobi_CUDA']

    df.sort_values(['Matrix_Size', 'Iterations'], inplace=True)

    for method in methods:
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        pivot = df.pivot(index='Matrix_Size', columns='Iterations', values=method)

        X = pivot.columns.values
        Y = pivot.index.values
        X_grid, Y_grid = np.meshgrid(X, Y)

        Z = moving_average(pivot, window_size=5).fillna(method='pad').values

        # Apply Gaussian smoothing
        Z = gaussian_filter(Z, sigma=1.0)

        surf = ax.plot_surface(X_grid, Y_grid, Z,
                               cmap='viridis', edgecolor='k', linewidth=0.3, antialiased=True)

        ax.set_xlabel('Iterations')
        ax.set_ylabel('Matrix Size')
        ax.set_zlabel('Time (μs)')
        ax.set_title(f'3D Plot - {method}')
        ax.set_zlim(0, 500_000)

        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='Execution Time (μs)')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/3d_{method}.png', dpi=150)
        plt.close()

    print(f'Generated 3D plots for: {", ".join(methods)}')

# 3D plots with different colors for every execution mode
def generate_3d_plot_with_colors(in_csv, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(in_csv)
    has_cuda = 'Gauss_CUDA' in df.columns and 'Jacobi_CUDA' in df.columns

    methods = ['Gauss_Serial', 'Gauss_OMP', 'Jacobi_Serial', 'Jacobi_OMP']
    if has_cuda:
        methods += ['Gauss_CUDA', 'Jacobi_CUDA']

    df.sort_values(['Matrix_Size', 'Iterations'], inplace=True)

    # Define color mapping
    color_map = {
        'Jacobi_Serial': '#ff9999',   # light red
        'Jacobi_OMP': '#ff4d4d',      # medium red
        'Jacobi_CUDA': '#b30000',     # dark red
        'Gauss_Serial': '#99ff99',    # light green
        'Gauss_OMP': '#33cc33',       # medium green
        'Gauss_CUDA': '#006600',      # dark green,
    }

    for method in methods:
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        pivot = df.pivot(index='Matrix_Size', columns='Iterations', values=method)

        X = pivot.columns.values
        Y = pivot.index.values
        X_grid, Y_grid = np.meshgrid(X, Y)

        Z = moving_average(pivot, window_size=5).fillna(method='pad').values
        # Z = gaussian_filter(Z, sigma=1.0)



        norm = plt.Normalize(Z.min(), Z.max())
        colors = plt.cm.get_cmap('RdYlGn_r')(norm(Z))
        surf = ax.plot_surface(
            X_grid, Y_grid, Z,
            facecolors=colors,
            edgecolor='k',
            linewidth=0.3,
            antialiased=True,
            alpha=0.9
        )


        ax.set_xlabel('Iterations')
        ax.set_ylabel('Matrix Size')
        ax.set_zlabel('Time (μs)')
        ax.set_title(f'3D Plot - {method}')
        ax.set_zlim(0, 500_000)

        plt.tight_layout()
        plt.savefig(f'{output_dir}/3d_{method}.png', dpi=150)
        plt.close()

    print(f"Generated colored 3D plots for: {', '.join(methods)}")


def plot_grouped_3d_planes(in_csv, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(in_csv)
    has_cuda = 'Gauss_CUDA' in df.columns and 'Jacobi_CUDA' in df.columns

    methods = ['Gauss_Serial', 'Gauss_OMP', 'Jacobi_Serial', 'Jacobi_OMP']
    if has_cuda:
        methods += ['Gauss_CUDA', 'Jacobi_CUDA']

    df.sort_values(['Matrix_Size', 'Iterations'], inplace=True)

    gauss_color_map = {
        'Gauss_Serial': '#ff9999',  # light red
        'Gauss_OMP': '#ffd966',     # light yellow
        'Gauss_CUDA': '#99e699'     # light green
    }

    jacobi_color_map = {
        'Jacobi_Serial': '#ff4d4d',  # medium red
        'Jacobi_OMP': '#ffcc00',     # medium yellow
        'Jacobi_CUDA': '#33cc33'     # medium green
    }

    orange_shades = {
        'Gauss_Serial': '#ffcc99',
        'Gauss_OMP': '#ff9966',
        'Gauss_CUDA': '#ff6600'
    }

    blue_shades = {
        'Jacobi_Serial': '#99ccff',
        'Jacobi_OMP': '#3399ff',
        'Jacobi_CUDA': '#0066cc'
    }

    def plot_surface_group(group_methods, color_map, filename, title):
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        for method in group_methods:
            if method not in df.columns:
                continue

            pivot = df.pivot(index='Matrix_Size', columns='Iterations', values=method)
            X = pivot.columns.values
            Y = pivot.index.values
            X_grid, Y_grid = np.meshgrid(X, Y)
            Z = moving_average(pivot, window_size=5).fillna(method='pad').values
            Z = gaussian_filter(Z, sigma=1.0)

            ax.plot_surface(
                X_grid, Y_grid, Z,
                color=color_map.get(method, 'gray'),
                edgecolor='k',
                linewidth=0.3,
                antialiased=True,
                alpha=0.7
            )

        ax.set_xlabel('Iterations')
        ax.set_ylabel('Matrix Size')
        ax.set_zlabel('Time (μs)')
        ax.set_title(title)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/{filename}', dpi=150)
        plt.close()

    # Plot 1: All Gauss
    plot_surface_group(
        ['Gauss_Serial', 'Gauss_OMP', 'Gauss_CUDA'] if has_cuda else ['Gauss_Serial', 'Gauss_OMP'],
        gauss_color_map,
        '3d_gauss.png',
        '3D Plot - Gauss Methods'
    )

    # Plot 2: All Jacobi
    plot_surface_group(
        ['Jacobi_Serial', 'Jacobi_OMP', 'Jacobi_CUDA'] if has_cuda else ['Jacobi_Serial', 'Jacobi_OMP'],
        jacobi_color_map,
        '3d_jacobi.png',
        '3D Plot - Jacobi Methods'
    )

    # Plot 3: All Serial
    plot_surface_group(
        ['Gauss_Serial', 'Jacobi_Serial'],
        {**orange_shades, **blue_shades},
        '3d_serial.png',
        '3D Plot - Serial (Gauss & Jacobi)'
    )

    # Plot 4: All OMP
    plot_surface_group(
        ['Gauss_OMP', 'Jacobi_OMP'],
        {**orange_shades, **blue_shades},
        '3d_omp.png',
        '3D Plot - OpenMP (Gauss & Jacobi)'
    )

    # Plot 5: All CUDA
    if has_cuda:
        plot_surface_group(
            ['Gauss_CUDA', 'Jacobi_CUDA'],
            {**orange_shades, **blue_shades},
            '3d_cuda.png',
            '3D Plot - CUDA (Gauss & Jacobi)'
        )

    print("Generated 3D plots grouped by method and implementation type.")



def plot_3d_overlay_planes_save(in_csv, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(in_csv)
    has_cuda = 'Gauss_CUDA' in df.columns and 'Jacobi_CUDA' in df.columns

    methods = ['Gauss_Serial', 'Gauss_OMP', 'Jacobi_Serial', 'Jacobi_OMP']
    if has_cuda:
        methods += ['Gauss_CUDA', 'Jacobi_CUDA']

    df.sort_values(['Matrix_Size', 'Iterations'], inplace=True)

    # Define color mapping
    color_map = {
        'Jacobi_Serial': '#ff9999',   # light red
        'Jacobi_OMP': '#ff4d4d',      # medium red
        'Jacobi_CUDA': '#b30000',     # dark red
        'Gauss_Serial': '#99ff99',    # light green
        'Gauss_OMP': '#33cc33',       # medium green
        'Gauss_CUDA': '#006600',      # dark green,
    }

    # Create a single figure for overlayed 3D plots
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    for method in methods:
        if method not in df.columns:
            continue

        pivot = df.pivot(index='Matrix_Size', columns='Iterations', values=method)

        X = pivot.columns.values
        Y = pivot.index.values
        X_grid, Y_grid = np.meshgrid(X, Y)

        Z = moving_average(pivot, window_size=5).fillna(method='pad').values
        Z = gaussian_filter(Z, sigma=1.0)

        # Plot the surface for each method with a distinct color
        ax.plot_surface(
            X_grid, Y_grid, Z,
            color=color_map.get(method, 'gray'),
            edgecolor='k',
            linewidth=0.3,
            antialiased=True,
            alpha=0.6  # Adjust alpha to ensure overlay visibility
        )

    # Set labels and title
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Matrix Size')
    ax.set_zlabel('Time (μs)')
    ax.set_title('Overlayed 3D Surface Plot (Gauss & Jacobi)')

    # ax.set_zlim(0, 500_000)

    # Tighten layout and save the figure
    plt.tight_layout()
    plt.savefig(f'{output_dir}/3d_overlay.png', dpi=150)
    plt.close()

    print(f"Generated overlayed 3D plot for: {', '.join(methods)}")


# 4D-like plot with bubbles and save as PNG
def plot_4d_like_bubbles_save(df, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    method_positions = {
        'Gauss_Serial': 0,
        'Gauss_OMP':    1,
        'Gauss_CUDA':   2,
        'Jacobi_Serial':3,
        'Jacobi_OMP':   4,
        'Jacobi_CUDA':  5
    }

    method_colors = {
        'Gauss_Serial':  'lightgreen',
        'Gauss_OMP':     'mediumseagreen',
        'Gauss_CUDA':    'darkgreen',
        'Jacobi_Serial': 'lightcoral',
        'Jacobi_OMP':    'indianred',
        'Jacobi_CUDA':   'darkred'
    }

    for method, z_pos in method_positions.items():
        if method not in df.columns:
            continue

        for matrix_size in df['Matrix_Size'].unique():
            subset = df[df['Matrix_Size'] == matrix_size]

            X = subset['Iterations'].values
            Y = np.full_like(X, matrix_size)
            Z = np.full_like(X, z_pos)
            sizes = subset[method].values

            # Normalize bubble size for visualization
            norm_sizes = 500 * (sizes - np.min(sizes)) / (np.max(sizes) - np.min(sizes) + 1e-9)

            ax.scatter(X, Y, Z, s=norm_sizes, alpha=0.6, c=method_colors[method], edgecolors='k', marker='o')

    ax.set_xlabel('Iterations')
    ax.set_ylabel('Matrix Size')
    ax.set_zlabel('Method')
    ax.set_zticks(list(method_positions.values()))
    ax.set_zticklabels(['Gauss_Serial', 'Gauss_OMP', 'Gauss_CUDA', 'Jacobi_Serial', 'Jacobi_OMP', 'Jacobi_CUDA'])
    ax.set_title('4D-like Bubble Plot (size = time)')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/4d_bubble_plot.png', dpi=150)
    plt.close()

if __name__ == "__main__":
    # CSV at ../cmake-build-release/omp_static_results.csv when generated by the CMake build
    generate_plots('omp_static_results.csv', 'omp_static')
    # generate_3d_plot('omp_static_results.csv', 'omp_static')
    generate_3d_plot_with_colors('omp_static_results.csv', 'omp_static')
    plot_grouped_3d_planes('omp_static_results.csv', 'omp_static')
    plot_3d_overlay_planes_save('omp_static_results.csv', 'omp_static')
    plot_4d_like_bubbles_save(pd.read_csv('omp_static_results.csv'), 'omp_static')
