# Real-Time Violence Detection System

## Overview
This system detects violent behavior in real-time video streams and triggers alerts when violence is detected. It combines YOLO for spatial analysis and LSTM for temporal analysis to achieve robust detection.

## System Architecture

### Components
1. **Video Processing Module**
   - Captures video frames
   - Runs YOLO classification
   - Processes sequences with LSTM

2. **Alert System**
   - Sends UDP alerts to server
   - Includes location and timestamp

3. **Server**
   - Receives alerts
   - Plays siren sound
   - Logs incident details

### Data Flow
```
Video Feed → YOLO → LSTM → Alert System → Server
```

## Technical Details

### Video Processing
- **Frame Rate**: 30 FPS
- **Processing Interval**: Every 5th frame
- **YOLO Model**: YOLOv8 classification
- **LSTM Sequence Length**: 10 predictions (~1.67 seconds)

### Network Communication
- **Protocol**: UDP
- **Port**: 9999
- **Message Format**: JSON
- **Message Size**: ~200 bytes

### Performance
- **Processing Latency**: ~200ms
- **Alert Transmission**: <1ms (local)
- **Total Response Time**: ~200-300ms

## Implementation Details

### Key Files
1. **RealtimeDetection.py**
   - Video capture and processing
   - YOLO and LSTM integration

2. **sendSignal.py**
   - Alert generation and transmission
   - Location and timestamp handling

3. **server.py**
   - Alert reception and handling
   - Siren sound playback

### Dependencies
- Python 3.8+
- PyTorch
- OpenCV
- Ultralytics YOLO
- Pygame

## System Characteristics

### Advantages
- Real-time processing
- Low latency alerts
- Cross-hardware compatibility
- Extensible architecture

### Limitations
- UDP packet loss possible
- Model accuracy depends on training data
- Limited to single camera in current implementation

## Potential Improvements
1. **Protocol Enhancements**
   - Add sequence numbers
   - Implement basic reliability

2. **Model Improvements**
   - Fine-tune on specific datasets
   - Experiment with architectures

3. **System Features**
   - Add authentication
   - Implement logging
   - Support multiple cameras

## Usage

### Running the System
1. Start the server:
   ```bash
   python server.py
   ```

2. Start the detection module:
   ```bash
   python sendSignal.py
   ```

### Configuration
- Edit `SECURITY_IP` in sendSignal.py for remote operation
- Adjust `frame_interval` and `seq_length` for performance tuning

## Hackathon Considerations

### Potential Questions
1. How does the system handle false positives/negatives?
2. What are the limitations of using UDP?
3. How can the system be scaled for multiple cameras?
4. What are the system's latency characteristics?
5. How does the model perform in different lighting conditions?

### Discussion Points
- Trade-offs between accuracy and latency
- Integration with existing security systems
- Potential applications beyond violence detection
