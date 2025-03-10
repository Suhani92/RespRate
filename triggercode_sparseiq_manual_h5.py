#this code connects to the radar via argparser
#and then records the sparse iq data for as much time as you want 
#and keeps giving updates regarding the amount of data recorded in multiple of 100 every 100 frames recorded
#in order to end the recording we need to end the session by using ctrl+c command and the file will be saved in the same directory as a h5 file 
from __future__ import annotations
import acconeer.exptool as et
from acconeer.exptool import a121
from acconeer.exptool.a121._core.entities.configs.config_enums import PRF, IdleState, Profile
from acconeer.exptool.a121.algo.sparse_iq import AmplitudeMethod, Processor, ProcessorConfig
import h5py
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

    # Create HDF5 file with timestamp in filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'sparse_iq_data_{timestamp}.h5'
    
    print(f"Recording data to {filename}")
    print("Press Ctrl-C to end session")

    with h5py.File(filename, 'w') as f:
        # Create groups for different types of data
        amplitude_group = f.create_group('amplitudes')
        velocity_group = f.create_group('velocity_maps')
        metadata_group = f.create_group('metadata')

        # Store configuration information
        metadata_group.attrs['timestamp'] = timestamp
        metadata_group.attrs['sensor_config'] = str(sensor_config)
        metadata_group.attrs['processor_config'] = str(processor_config)
        
        # Create extensible datasets for storing the measurements
        amplitude_dataset = amplitude_group.create_dataset(
            'amplitude_data',
            shape=(0, sensor_config.subsweeps[2].num_points),  # For third subsweep
            maxshape=(None, sensor_config.subsweeps[2].num_points),
            dtype=np.float32
        )
        
        # We'll determine velocity map shape from the first measurement
        first_frame = True
        frame_count = 0
        velocity_dataset = None
        
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
                
                # Create velocity dataset once we know its shape
                if first_frame:
                    velocity_shape = distance_velocity_map.shape
                    velocity_dataset = velocity_group.create_dataset(
                        'velocity_data',
                        shape=(0,) + velocity_shape,
                        maxshape=(None,) + velocity_shape,
                        dtype=np.float32
                    )
                    first_frame = False
                
                # Store the amplitudes
                current_size = amplitude_dataset.shape[0]
                amplitude_dataset.resize(current_size + 1, axis=0)
                amplitude_dataset[current_size] = result_third_subsweep.amplitudes
                
                # Store the velocity map
                velocity_dataset.resize(current_size + 1, axis=0)
                velocity_dataset[current_size] = distance_velocity_map
                
                frame_count += 1
                if frame_count % 100 == 0:
                    duration = time.time() - start_time
                    print(f"Recorded {frame_count} frames ({frame_count/duration:.1f} fps)")

        except KeyboardInterrupt:
            print("\nRecording stopped by user")
        except Exception as e:
            print(f"An error occurred: {e}")
        finally:
            # Store final metadata
            metadata_group.attrs['total_frames'] = frame_count
            metadata_group.attrs['duration'] = time.time() - start_time
            metadata_group.attrs['average_fps'] = frame_count / (time.time() - start_time)
            
            print(f"\nRecording complete:")
            print(f"- Total frames: {frame_count}")
            print(f"- Data saved to: {filename}")
            print(f"- Average frame rate: {frame_count/(time.time() - start_time):.1f} fps")

    print("Disconnecting...")
    client.close()

if __name__ == "__main__":
    main()