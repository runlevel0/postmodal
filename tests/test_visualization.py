import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from postmodal.complexity import calculate_complexity_metrics
from postmodal.visualization import (
    plot_mac_matrix,
    plot_modeshape_complexity,
    plot_modeshape_complexity_grid,
)

matplotlib.use("Agg")  # Use non-interactive backend for testing


class TestVisualization:
    def setup_method(self) -> None:
        """Set up test data."""
        # Create a MAC matrix
        self.mac_matrix = np.array([[1.0, 0.2, 0.1], [0.2, 1.0, 0.3], [0.1, 0.3, 1.0]])

        # Create modeshapes for complexity plotting
        self.real_modeshape = np.array([1.0, 2.0, 3.0, 4.0])
        self.complex_modeshape = np.array([1.0 + 0.5j, 2.0 + 1.0j, 3.0 + 1.5j, 4.0 + 2.0j])
        self.highly_complex_modeshape = np.array([1.0 + 1.0j, -2.0 + 2.0j, -3.0 - 3.0j, 4.0 - 4.0j])

        # Create a batch of modeshapes
        self.modeshape_batch = np.array([self.real_modeshape, self.complex_modeshape, self.highly_complex_modeshape])

        # Calculate complexity metrics for the batch
        self.complexity_metrics = calculate_complexity_metrics(self.modeshape_batch)

    def teardown_method(self) -> None:
        """Close all matplotlib figures after each test."""
        plt.close("all")

    def test_plot_mac_matrix(self) -> None:
        """Test MAC matrix plotting."""
        # Test with default parameters
        x_labels = ["Mode 1", "Mode 2", "Mode 3"]
        y_labels = ["Test 1", "Test 2", "Test 3"]

        fig, ax = plot_mac_matrix(self.mac_matrix, x_tick_labels=x_labels, y_tick_labels=y_labels)

        # Check that the function returns a figure and axis
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)

        # Test with custom parameters
        fig, ax = plot_mac_matrix(
            self.mac_matrix,
            x_tick_labels=x_labels,
            y_tick_labels=y_labels,
            text_color_variable=False,
            invert_scale=True,
        )

        # Check that the function returns a figure and axis
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)

    def test_plot_modeshape_complexity(self) -> None:
        """Test modeshape complexity plotting."""
        # Test with a single modeshape
        fig, ax = plot_modeshape_complexity(self.complex_modeshape)

        # Check that the function returns a figure and axis
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)

    def test_plot_modeshape_complexity_grid(self) -> None:
        """Test modeshape complexity grid plotting."""
        # Create test frequencies
        frequencies = np.array([1.0, 2.0, 3.0])

        # Test with a batch of modeshapes
        fig, axes = plot_modeshape_complexity_grid(frequencies=frequencies, modeshapes=self.modeshape_batch)

        # Check that the function returns a figure and axes
        assert isinstance(fig, plt.Figure)
        assert len(axes.flatten()) >= len(self.modeshape_batch)

        # Test with custom parameters
        fig, axes = plot_modeshape_complexity_grid(
            frequencies=frequencies,
            modeshapes=self.modeshape_batch,
            figsize=(15, 5),
            n_row=1,
            n_col=3,
            hspace=0.3,
            wspace=0.3,
        )

        # Check that the function returns a figure and axes
        assert isinstance(fig, plt.Figure)
        assert len(axes.flatten()) >= len(self.modeshape_batch)
