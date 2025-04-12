import numpy as np
import pandas as pd
from scipy.linalg import hankel, svd, eig
from numpy.typing import NDArray
import numpy as np
from scipy import signal
from scipy.linalg import hankel, svd, eig
from collections import OrderedDict
from typing import Tuple,Dict
from dataclasses import dataclass
from tqdm import tqdm # type: ignore
import gc
from matplotlib import pyplot as plt
import plotly.graph_objects as go

def construct_and_svd_block_toeplitz(impulse_response_function: NDArray) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
    """
    Constructs a block Toeplitz matrix from the given impulse response function and performs SVD.

    Args:
        impulse_response_function: Impulse Response Function as a NumPy array.

    Returns:
        A tuple containing:
            U: Left singular vectors from SVD.
            S: Singular values from SVD.
            V: Right singular vectors from SVD.
            block_toeplitz_matrix: The constructed block Toeplitz matrix.

    Assumptions:
        - impulse_response_function has shape (:, :, n) where n is odd.
    """
    number_channels = impulse_response_function.shape[0]
    irf_length = impulse_response_function.shape[2]
    block_size = number_channels
    num_blocks = irf_length - 1
    
    # Initialize lists to store blocks
    blocks = []

    # Progress bar for Toeplitz matrix consrtruction
    with tqdm(total=num_blocks * num_blocks, desc="Building Toeplitz Matrix", unit="block") as pbar:
        for i in range(num_blocks):
            row_blocks = []
            for j in range(num_blocks):
                if i >= j:
                    row_blocks.append(impulse_response_function[:, :, i - j])
                else:
                    row_blocks.append(np.zeros_like(impulse_response_function[:, :, 0]))
                pbar.update(1)
            blocks.append(row_blocks)
    # Concatenate blocks to form the block Toeplitz matrix
    block_toeplitz_matrix = np.block(blocks)
    del blocks  # Free up memory
    gc.collect()
    
    # Perform Singular Value Decomposition
    U, S, V = svd(block_toeplitz_matrix)
    gc.collect()
    
    return U, S, V, block_toeplitz_matrix


@dataclass
class SSICov:
    """
    Implements the Stochastic Subspace Identification with Covariance (SSICOV) method for modal identification.

    Args:
        df: DataFrame containing time and acceleration data.
        acceleration_labels: Label for the acceleration column in the DataFrame.
        min_model_order: Minimum model order to consider. Defaults to 2.
        max_model_order: Maximum model order to consider. Defaults to 32.

    Returns:
        None

    Assumptions:
        - The DataFrame contains columns named "{label} Acceleration" for each label in the 'acceleration_labels' list.
        - max_model_order is greater than min_model_order.
    """

    df: pd.DataFrame
    acceleration_labels: list
    min_model_order: int = 2
    max_model_order: int = 32

    def __post_init__(self):
        """
        Post initialization to validate input parameters and calculate sampling rate.

        Raises:
            ValueError: If max_model_order is not greater than min_model_order, or if required columns are missing.
        """
        if self.max_model_order <= self.min_model_order:
            raise ValueError("max_model_order must be greater than min_model_order")

        self.delta_t = self.df['Time'].iloc[1] - self.df['Time'].iloc[0]
        self.fs = 1 / self.delta_t
        self.total_time = self.df['Time'].iloc[-1] - self.df['Time'].iloc[0]
        self.number_samples = int(round(self.total_time * self.fs))
        self.number_channels = len(self.acceleration_labels)
        self.df_acceleration = pd.DataFrame()
        for channel, label in enumerate(self.acceleration_labels):
            self.df_acceleration[f'Channel {channel}'] = self.df[f'{label} Acceleration'].values

    def compute_impulse_response_function(self) -> NDArray:
        """
        Computes the impulse response function using FFT.

        Args:
            None

        Returns:
            Impulse response function as a NumPy array.

        Assumptions:
            - The acceleration data is uniformly sampled.
        """
        delta_t = 1 / self.fs
        number_samples = round(self.total_time / delta_t)
        impulse_response_function = np.zeros((self.number_channels, self.number_channels, number_samples - 1), dtype=complex)

        # Bar progress for IRF computation
        with tqdm(total=self.number_channels * self.number_channels, desc="Computing IRF", unit="channel pair") as pbar:
            for channel_1 in range(self.number_channels):
                for channel_2 in range(self.number_channels):
                    fft_channel_1 = np.fft.fft(self.df_acceleration[f'Channel {channel_1}'])
                    fft_channel_2 = np.fft.fft(self.df_acceleration[f'Channel {channel_2}'])
                    cross_spectrum = np.fft.ifft(fft_channel_1 * fft_channel_2.conj())
                    impulse_response_function[channel_1, channel_2, :] = np.real(cross_spectrum[0:number_samples - 1])
                    pbar.update(1)

        if self.number_channels == 1:
            impulse_response_function = np.squeeze(impulse_response_function)
            impulse_response_function = impulse_response_function / abs(impulse_response_function[0])

        return impulse_response_function

    def construct_block_toeplitz(self, impulse_response_function: NDArray) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
        """
        Constructs a block Toeplitz matrix from the given impulse response function and performs SVD.

        Args:
            impulse_response_function: Impulse Response Function as a NumPy array.

        Returns:
            A tuple containing:
                U: Left singular vectors from SVD.
                S: Singular values from SVD.
                V: Right singular vectors from SVD.
                block_toeplitz_matrix: The constructed block Toeplitz matrix.

        Assumptions:
            - impulse_response_function has shape (:, :, n) where n is odd.
        """
        return construct_and_svd_block_toeplitz(impulse_response_function)

    def identify_modal_parameters(self, left_singular_vectors: NDArray, singular_values: NDArray, num_modes: int, num_sensors: int, sampling_frequency: float) -> Tuple[NDArray, NDArray, NDArray]:
        """
        Identifies modal parameters (natural frequencies, damping ratios, and mode shapes) from SVD results.

        Args:
            left_singular_vectors: Left singular vectors from SVD.
            singular_values: Singular values from SVD.
            num_modes: Number of modes to identify.
            num_sensors: Number of sensors.
            sampling_frequency: Sampling frequency.

        Returns:
            A tuple containing:
                natural_frequencies: Natural frequencies.
                damping_ratios: Damping ratios.
                mode_shapes: Mode shapes.

        Assumptions:
            - left_singular_vectors and singular_values are outputs from Singular Value Decomposition (SVD).
            - num_modes is a positive integer.
            - num_sensors is less than or equal to the number of rows in left_singular_vectors.
            - sampling_frequency is the sampling frequency in Hz.
        """
        singular_values_diag = np.diag(singular_values)

        if num_modes >= singular_values_diag.shape[0]:
            print("changing the number of modes to the maximum possible")
            num_modes = singular_values_diag.shape[0]

        time_step = 1 / sampling_frequency
        observability_matrix = np.matmul(left_singular_vectors[:, 0:num_modes], np.sqrt(singular_values_diag[0:num_modes, 0:num_modes]))
        index_observability = min(num_sensors, observability_matrix.shape[0])
        C_matrix = observability_matrix[0:index_observability, :]
        num_blocks = observability_matrix.shape[0] / index_observability
        ao = int((index_observability) * (num_blocks - 1))
        bo = int(len(observability_matrix[:, 0]) - (index_observability) * (num_blocks - 1))
        co = len(observability_matrix[:, 0])
        state_matrix = np.matmul(np.linalg.pinv(observability_matrix[0:ao, :]), observability_matrix[bo:co, :])
        eigenvalues, eigenvectors = np.linalg.eig(state_matrix)
        mu = np.log(np.diag(np.diag(eigenvalues))) / time_step
        natural_frequencies_all = np.abs(mu) / (2 * np.pi)
        natural_frequencies = natural_frequencies_all[np.ix_(*[range(0, i, 2) for i in natural_frequencies_all.shape])]
        damping_ratios_all = -np.real(mu) / np.abs(mu)
        damping_ratios = damping_ratios_all[np.ix_(*[range(0, i, 2) for i in damping_ratios_all.shape])]
        mode_shapes_all = np.real(np.matmul(C_matrix[0:index_observability, :], eigenvectors))
        mode_shapes = mode_shapes_all[:, 1::2]

        return natural_frequencies, damping_ratios, mode_shapes

    def perform_stability_checks(self, prev_frequencies: NDArray, prev_damping_ratios: NDArray, prev_mode_shapes: NDArray, curr_frequencies: NDArray, curr_damping_ratios: NDArray, curr_mode_shapes: NDArray) -> Tuple[NDArray, NDArray, NDArray, NDArray, NDArray]:
        """
        Performs stability checks based on frequency, damping, and mode shape.

        Args:
            prev_frequencies: Natural frequencies from the previous model order.
            prev_damping_ratios: Damping ratios from the previous model order.
            prev_mode_shapes: Mode shapes from the previous model order.
            curr_frequencies: Natural frequencies from the current model order.
            curr_damping_ratios: Damping ratios from the current model order.
            curr_mode_shapes: Mode shapes from the current model order.

        Returns:
            A tuple containing:
                stable_frequencies: Stable natural frequencies.
                stable_damping_ratios: Stable damping ratios.
                stable_mode_shapes: Stable mode shapes.
                MAC: Mode shape correlation values.
                stability_status: Stability status flags.

        Assumptions:
            - Input modal parameters are from models of adjacent orders.
        """
        eps_freq = 1e-2  # Umbral original de variación de frecuencia
        eps_zeta = 4e-2
        eps_MAC = 1e-2  # Umbral original de variación de MAC
        stability_status = []
        stable_frequencies = []
        stable_damping_ratios = []
        stable_mode_shapes_list = []
        MAC = []

        num_prev_frequencies = len(prev_frequencies)
        num_curr_frequencies = len(curr_frequencies)

        for prev_index in range(num_prev_frequencies):
            for curr_index in range(num_curr_frequencies):
                # Ensure indices are within bounds
                if prev_index >= prev_mode_shapes.shape[1] or curr_index >= curr_mode_shapes.shape[1]:
                    continue

                freq_stability = self.check_relative_difference(prev_frequencies[prev_index], curr_frequencies[curr_index], eps_freq)
                damping_stability = self.check_relative_difference(prev_damping_ratios[prev_index], curr_damping_ratios[curr_index], eps_zeta)
                mode_shape_stability, mac_value = self.compute_MAC(prev_mode_shapes[:, prev_index], curr_mode_shapes[:, curr_index], eps_MAC)

                if freq_stability == 0:
                    stability_status_flag = 0  # new pole
                elif freq_stability == 1 and mode_shape_stability == 1 and damping_stability == 1:
                    stability_status_flag = 1  # stable pole
                elif freq_stability == 1 and damping_stability == 0 and mode_shape_stability == 1:
                    stability_status_flag = 2  # pole with stable frequency and vector
                elif freq_stability == 1 and damping_stability == 1 and mode_shape_stability == 0:
                    stability_status_flag = 3  # pole with stable frequency and damping
                elif freq_stability == 1 and damping_stability == 0 and mode_shape_stability == 0:
                    stability_status_flag = 4  # pole with stable frequency
                else:
                    raise ValueError("Error: stability_status is undefined")

                stable_frequencies.append(curr_frequencies[curr_index])
                stable_damping_ratios.append(curr_damping_ratios[curr_index])
                stable_mode_shapes_list.append(curr_mode_shapes[:, curr_index])
                MAC.append(mac_value)
                stability_status.append(stability_status_flag)

        sorted_indices = np.argsort(stable_frequencies)
        stable_frequencies = np.sort(stable_frequencies)
        stable_damping_ratios = np.array(stable_damping_ratios)[sorted_indices]
        stable_mode_shapes = np.column_stack(stable_mode_shapes_list)[:, sorted_indices]
        MAC = np.array(MAC)[sorted_indices]
        stability_status = np.array(stability_status)[sorted_indices]

        return stable_frequencies, stable_damping_ratios, stable_mode_shapes, MAC, stability_status

    def check_relative_difference(self, value1: float, value2: float, tolerance: float) -> int:
        """
        Checks if the relative difference between two values is within a tolerance.

        Args:
            value1: First value.
            value2: Second value.
            tolerance: Tolerance.

        Returns:
            1 if the relative difference is within the tolerance, 0 otherwise.

        Assumptions:
            - Inputs are scalar values.
        """
        return 1 if abs(1 - value1 / value2) < tolerance else 0

    def compute_MAC(self, mode_shape1: NDArray, mode_shape2: NDArray, tolerance: float) -> Tuple[int, float]:
        """
        Computes the Mode Assurance Criterion (MAC) between two mode shapes.

        Args:
            mode_shape1: First mode shape.
            mode_shape2: Second mode shape.
            tolerance: Tolerance for MAC value.

        Returns:
            A tuple containing:
                is_above_threshold: 1 if MAC is above the threshold, 0 otherwise.
                mac_value: MAC value.

        Assumptions:
            - Input mode shapes are vectors.
        """
        numerator = np.abs(np.dot(mode_shape1.flatten(), mode_shape2.flatten())) ** 2
        denominator1 = np.dot(mode_shape1.flatten(), mode_shape1.flatten())
        denominator2 = np.dot(mode_shape2.flatten(), mode_shape2.flatten())
        mac_value = numerator / (denominator1 * denominator2)
        is_above_threshold = 1 if mac_value > (1 - tolerance) else 0

        return is_above_threshold, mac_value

    def reverse_dictionary(self, input_dict: Dict) -> OrderedDict:
        """
        Reverses the order of elements in a dictionary.

        Args:
            input_dict: Input dictionary.

        Returns:
            reversed_dict: Reversed dictionary.

        Assumptions:
            - Input is a dictionary.
        """
        ordered_dict = OrderedDict(input_dict)
        reversed_dict = OrderedDict()
        for key in reversed(ordered_dict):
            reversed_dict[key] = ordered_dict[key]

        return reversed_dict

    def extract_stable_poles(self, frequencies: Dict[int, NDArray], damping_ratios: Dict[int, NDArray], mode_shapes: Dict[int, NDArray], MAC_values: Dict[int, NDArray], stability_statuses: Dict[int, NDArray]) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
        """
        Extracts stable poles based on the stability status.

        Args:
            frequencies: Natural frequencies.
            damping_ratios: Damping ratios.
            mode_shapes: Mode shapes.
            MAC_values: Mode shape correlation values.
            stability_statuses: Stability status flags.

        Returns:
            A tuple containing:
                stable_frequencies: Stable natural frequencies.
                stable_damping_ratios: Stable damping ratios.
                stable_mode_shapes: Stable mode shapes.
                stable_MAC_values: MAC values for stable poles.

        Assumptions:
            - Input modal parameters and stability statuses are properly aligned.
        """
        stable_frequencies = []
        stable_damping_ratios = []
        stable_mode_shapes = []
        stable_MAC_values = []

        for i in range(len(frequencies)):
            for j in range(len(stability_statuses[i])):
                if stability_statuses[i][j] == 1:
                    stable_frequencies.append(frequencies[i][j])
                    stable_damping_ratios.append(damping_ratios[i][j])
                    stable_mode_shapes.append(mode_shapes[i][:, j])
                    stable_MAC_values.append(MAC_values[i][j])

        stable_frequencies = np.array(stable_frequencies)
        stable_damping_ratios = np.array(stable_damping_ratios)
        stable_mode_shapes = np.array(stable_mode_shapes).T
        stable_MAC_values = np.array(stable_MAC_values)

        valid_indices = stable_damping_ratios > 0
        stable_frequencies = stable_frequencies[valid_indices]
        stable_mode_shapes = stable_mode_shapes[:, valid_indices]
        stable_MAC_values = stable_MAC_values[valid_indices]
        stable_damping_ratios = stable_damping_ratios[valid_indices]

        for index in range(stable_mode_shapes.shape[1]):
            stable_mode_shapes[:, index] = stable_mode_shapes[:, index] / np.max(np.abs(stable_mode_shapes[:, index]))
            if np.diff(stable_mode_shapes[0:2, index]) < 0:
                stable_mode_shapes[:, index] = -stable_mode_shapes[:, index]

        return stable_frequencies, stable_damping_ratios, stable_mode_shapes, stable_MAC_values

    def perform_stability_analysis(self, U: NDArray, S: NDArray) -> Tuple[Dict[int, NDArray], Dict[int, NDArray], Dict[int, NDArray], Dict[int, NDArray], Dict[int, NDArray]]:
        """
        Performs the stability analysis over a range of model orders.

        Args:
            U: Left singular vectors from SVD.
            S: Singular values from SVD.

        Returns:
            A tuple containing:
                natural_frequencies: Natural frequencies for each model order.
                damping_ratios: Damping ratios for each model order.
                mode_shapes: Mode shapes for each model order.
                MAC_values: MAC values for each model order.
                stability_statuses: Stability status for each model order.

        Assumptions:
            - U and S are the outputs of a Singular Value Decomposition (SVD).
        """
        natural_frequencies, damping_ratios, mode_shapes, MAC_values, stability_statuses = {}, {}, {}, {}, {}
        prev_frequencies, prev_damping_ratios, prev_mode_shapes = None, None, None
        
        with tqdm(total=self.max_model_order - self.min_model_order + 1, desc="Performing Stability Analysis", unit="model order") as pbar:
            for iteration, model_order in enumerate(range(self.max_model_order, self.min_model_order - 1, -1)):
                curr_frequencies, curr_damping_ratios, curr_mode_shapes = self.identify_modal_parameters(U, S, model_order, self.number_channels, self.fs)

                if iteration > 0:
                    stable_frequencies, stable_damping_ratios, stable_mode_shapes, stable_MAC_values, stability_status = self.perform_stability_checks(
                        prev_frequencies, prev_damping_ratios, prev_mode_shapes, curr_frequencies, curr_damping_ratios, curr_mode_shapes
                    )

                    natural_frequencies[iteration - 1] = stable_frequencies
                    damping_ratios[iteration - 1] = stable_damping_ratios
                    mode_shapes[iteration - 1] = stable_mode_shapes
                    MAC_values[iteration - 1] = stable_MAC_values
                    stability_statuses[iteration - 1] = stability_status

                prev_frequencies, prev_damping_ratios, prev_mode_shapes = curr_frequencies, curr_damping_ratios, curr_mode_shapes
                pbar.update(1)

        return natural_frequencies, damping_ratios, mode_shapes, MAC_values, stability_statuses

    def execute_ssicov_analysis(self) -> Tuple[NDArray, NDArray, NDArray, NDArray, Dict[int, NDArray], Dict[int, NDArray]]:
        """
        Executes the complete SSICOV analysis.

        Args:
            None

        Returns:
            A tuple containing:
                stable_frequencies: Stable natural frequencies.
                stable_damping_ratios: Stable damping ratios.
                stable_mode_shapes: Stable mode shapes.
                stable_MAC_values: MAC values for stable poles.
                stability_statuses: Stability status for each model order.
                natural_frequencies: Natural frequencies for each model order.

        Assumptions:
            - Input data and parameters have been properly initialized.
        """
        impulse_response_function = self.compute_impulse_response_function()
        U, S, V, _ = self.construct_block_toeplitz(impulse_response_function)
        natural_frequencies, damping_ratios, mode_shapes, MAC_values, stability_statuses = self.perform_stability_analysis(U, S)
        natural_frequencies = self.reverse_dictionary(natural_frequencies)
        damping_ratios = self.reverse_dictionary(damping_ratios)
        mode_shapes = self.reverse_dictionary(mode_shapes)
        MAC_values = self.reverse_dictionary(MAC_values)
        stability_statuses = self.reverse_dictionary(stability_statuses)
        stable_frequencies, stable_damping_ratios, stable_mode_shapes, stable_MAC_values = self.extract_stable_poles(
            natural_frequencies, damping_ratios, mode_shapes, MAC_values, stability_statuses
        )

        return stable_frequencies, stable_damping_ratios, stable_mode_shapes, stable_MAC_values, stability_statuses, natural_frequencies

    def calculate_cpsd(
        self,
        acceleration_data: np.ndarray,
        sampling_frequency: float,
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Calculate the Cross Power Spectral Density (CPSD) of the acceleration data.

        Args:
            acceleration_data: 2D array of acceleration data.
            sampling_frequency: Sampling frequency of the data.

        Returns:
            Tuple containing:
            - frequency_axis_id: Array of frequency values.
            - trace_spectrum_xx: Array of CPSD values.
            - N: Number of segments used in the CPSD calculation.
        """
        nperseg = min(2048, acceleration_data.shape[0])
        frequency_axis_id, trace_spectrum_xx = signal.csd(
            acceleration_data[:, 0], acceleration_data[:, 1], fs=sampling_frequency, nperseg=nperseg
        )
        N = len(frequency_axis_id)
        return frequency_axis_id, trace_spectrum_xx, N


    def plot_stability_diagram(self):
        """
        Generates a stabilization diagram plot overlaid with CPSD.

        Returns:
            None
        """
        stable_frequencies, stable_damping_ratios, stable_mode_shapes, stable_MAC_values, stability_statuses, natural_frequencies = self.execute_ssicov_analysis()

        sampling_frequency = 1 / (self.df["Time"].iloc[1] - self.df["Time"].iloc[0])
        # number_channels = len(acceleration_labels) #Variable is not used

        # Clarity: Convert acceleration data to NDArray
        df_acceleration = self.df[[f"{label} Acceleration" for label in self.acceleration_labels]].values

        nperseg = min(2048, df_acceleration.shape[0])

        frequency_axis_id, trace_spectrum_xx, N = self.calculate_cpsd(
            df_acceleration,
            sampling_frequency,
        )

        model_orders = np.arange(self.min_model_order, self.max_model_order + 1)
        fig, ax1 = plt.subplots()

        # Plot stability_status
        markers = ["k+", "ro", "bo", "gs", "gx"]
        labels = [
            "new pole",
            "stable pole",
            "stable freq. & MAC",
            "stable freq. & damp.",
            "stable freq.",
        ]
        handles = []
        for jj in range(5):
            x = []
            y = []
            for ii in tqdm(range(len(natural_frequencies)), desc=f"Processing status {jj+1}/5"):  # Algo nuevo: Progress bar
                try:
                    ind = np.where(stability_statuses[ii] == jj)
                    x.extend(
                        natural_frequencies[ii][ind].flatten()
                    )  # Flatten the array to ensure it's iterable
                    y.extend([model_orders[ii]] * len(natural_frequencies[ii][ind]))
                except Exception as e:
                    print(f"Error processing stability status: {e}")
            h, = ax1.plot(x, y, markers[jj], label=labels[jj])
            handles.append(h)

        ax1.set_ylabel("number of poles")
        ax1.set_xlabel("f (Hz)")
        ax1.set_ylim(0, self.max_model_order + 2)
        ax2 = ax1.twinx()

        max_freq = 0
        for freqs in natural_frequencies.values():
            if (
                len(freqs) > 0
            ):  # Check if the array is not empty, handling cases where natural frequencies could be empty
                max_freq = max(max_freq, np.max(freqs))

        # Plot CPSD
        color = "blue"
        ax2.set_xlabel("frequency [Hz]")
        ax2.set_ylabel("Power Spectral Density", color=color)
        ax2.plot(frequency_axis_id, 10 * np.log10(trace_spectrum_xx / N), color, label="Trace")
        ax2.tick_params(axis="y", labelcolor=color)
        ax2.set_xlim(0, max_freq * 2)
        # ax2.set_yscale('log') # commented out because log scale could hide details

        ax1.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=5)

        fig.tight_layout(rect=[0, 0.1, 1, 1])
        plt.show()

    def plotly_stability_diagram(self) -> go.Figure:
        """
        Generates a stabilization diagram plot overlaid with CPSD using Plotly.

        Returns:
            The Plotly figure.
        """
        stable_frequencies, stable_damping_ratios, stable_mode_shapes, stable_MAC_values, stability_statuses, natural_frequencies = self.execute_ssicov_analysis()

        sampling_frequency = 1 / (self.df["Time"].iloc[1] - self.df["Time"].iloc[0])
        # number_channels = len(acceleration_labels) #Variable is not used

        # Clarity: Convert acceleration data to NDArray
        df_acceleration = self.df[[f"{label} Acceleration" for label in self.acceleration_labels]].values

        nperseg = min(2048, df_acceleration.shape[0])

        frequency_axis_id, trace_spectrum_xx, N = self.calculate_cpsd(
            df_acceleration,
            sampling_frequency,
        )

        model_orders = np.arange(self.min_model_order, self.max_model_order + 1)

        fig = go.Figure()

        # Plot stability_status
        markers = ["circle", "circle", "circle", "circle", "circle"]  # Using different shapes
        colors = ["black", "red", "blue", "green", "purple"]  # Defining colors
        labels = [
            "new pole",
            "stable pole",
            "stable freq. & MAC",
            "stable freq. & damp.",
            "stable freq.",
        ]

        max_pole_frequency = 0  # Variable to track the maximum frequency identified by poles

        for jj in range(5):
            x = []
            y = []
            for ii in tqdm(range(len(natural_frequencies)), desc=f"Processing status {jj+1}/5"):  # Algo nuevo: Progress bar
                try:
                    ind = np.where(stability_statuses[ii] == jj)
                    x.extend(natural_frequencies[ii][ind].flatten())  # Flatten the array to ensure it's iterable
                    y.extend([model_orders[ii]] * len(natural_frequencies[ii][ind]))
                    if len(natural_frequencies[ii][ind]) > 0:
                        max_pole_frequency = max(max_pole_frequency, np.max(natural_frequencies[ii][ind]))
                except Exception as e:
                    print(f"Error processing stability status: {e}")

            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="markers",
                    marker=dict(symbol=markers[jj], color=colors[jj]),
                    name=labels[jj],
                )
            )

        # Plot CPSD
        fig.add_trace(
            go.Scatter(
                x=frequency_axis_id,
                y=10 * np.log10(np.abs(trace_spectrum_xx) / N),  # Use the magnitude of the complex values
                mode="lines",
                name="Trace CPSD",
                yaxis="y2",  # Use the second y-axis for CPSD
            )
        )

        # Configure layout with dual y-axes
        fig.update_layout(
            title="Stability Diagram Overlaid with CPSD",
            xaxis_title="Frequency (Hz)",
            yaxis_title="Model Order",
            xaxis=dict(range=[0, max_pole_frequency * 1.1]),  # Set x-axis limit to 110% of the maximum pole frequency
            yaxis2=dict(title="Power Spectral Density (dB)", overlaying="y", side="right"),
            template="plotly_white",
        )

        fig.write_html("results/stability_diagram.html")

        return fig