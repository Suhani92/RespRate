#this code triggers the radar and connects to it using argparser and starts recording the iq data 
#the recorded iq data is then stored in a csv file after we manuall press ctrl + C to interrupt the process
#and the it is displayed in the terminal every 100 frames that how many frames got 
#saved in the same directory automatically
from __future__ import annotations
import acconeer.exptool as et
from acconeer.exptool import a121
from acconeer.exptool.a121._core.entities.configs.config_enums import PRF, IdleState, Profile
from acconeer.exptool.a121.algo.sparse_iq import AmplitudeMethod, Processor, ProcessorConfig
import csv
import numpy as np
import time
import datetime

def main():
    # Parse arguments and set up logging
    args = a121.ExampleArgumentParser().parse_args()
    et.utils.config_logging(args)

    # Connect to the radar
    client = a121.Client.open(**a121.get_client_args(args))

    # Set up processor configuration
    processor_config = ProcessorConfig()
    processor_config.amplitude_method = AmplitudeMethod.COHERENT
    sensor_id = 1

    # Configure the sensor with multiple subsweeps
    sensor_config = a121.SensorConfig(
        sweeps_per_frame=8,
        sweep_rate=None,
        frame_rate=None,
        inter_frame_idle_state=IdleState.READY,
        inter_sweep_idle_state=IdleState.READY,
        continuous_sweep_mode=False,
        double_buffering=False,
        subsweeps=[
            a121.SubsweepConfig(start_point=70),
            a121.SubsweepConfig(),
            a121.SubsweepConfig(profile=Profile.PROFILE_2),
        ],
    )

    # Configure additional subsweep parameters
    sensor_config.subsweeps[0].num_points = 140
    sensor_config.subsweeps[1].prf = PRF.PRF_13_0_MHz

    # Create session configuration with multiple groups
    session_config = a121.SessionConfig(
        [
            {sensor_id: sensor_config},
            {sensor_id: a121.SensorConfig(sweeps_per_frame=20)},
        ],
        extended=True,
    )

    # Setup and start the session
    client.setup_session(session_config)
    client.start_session()
    processor = Processor(session_config=session_config, processor_config=processor_config)

    # Create CSV file with timestamp in filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'sparse_iq_data_{timestamp}.csv'

    print(f"Recording data to {filename}")
    print("Press Ctrl-C to end session")

    # Open CSV file and write headers
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)

        # Write metadata
        writer.writerow(["timestamp", timestamp])
        writer.writerow(["sensor_config", str(sensor_config)])
        writer.writerow(["processor_config", str(processor_config)])
        writer.writerow([])  # Empty line to separate metadata from data

        # Write data headers
        writer.writerow(["frame_index", "amplitude_data", "velocity_data"])

        frame_count = 0
        interrupt_handler = et.utils.ExampleInterruptHandler()
        start_time = time.time()

        try:
            while not interrupt_handler.got_signal:
                results = client.get_next()
                result_sensor_configs = processor.process(results=results)
                result_first_sensor_config = result_sensor_configs[0][sensor_id]
                result_third_subsweep = result_first_sensor_config[2]

                # Get the distance velocity map from the second group
                distance_velocity_map = result_sensor_configs[1][sensor_id][0].distance_velocity_map

                # Write data to the CSV file
                amplitude_data = result_third_subsweep.amplitudes.tolist()
                velocity_data = distance_velocity_map.flatten().tolist()
                writer.writerow([frame_count, amplitude_data, velocity_data])

                frame_count += 1
                if frame_count % 100 == 0:
                    duration = time.time() - start_time
                    print(f"Recorded {frame_count} frames ({frame_count/duration:.1f} fps)")

        except KeyboardInterrupt:
            print("\nRecording stopped by user")
        except Exception as e:
            print(f"An error occurred: {e}")
        finally:
            print(f"\nRecording complete:")
            print(f"- Total frames: {frame_count}")
            print(f"- Data saved to: {filename}")
            print(f"- Average frame rate: {frame_count/(time.time() - start_time):.1f} fps")

    print("Disconnecting...")
    client.close()

if __name__ == "__main__":
    main()
