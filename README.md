# Music_AI_Model
Overview: Real-time Music generation controlled by hand movements. Model was trained on public-domain Chopin piano pieces. 


The project - 
Details: MediaPipe  was used for hand tracking, PyTorch LSTM trained on Chopin MIDI dataset. Left hand controls creativity level, while right hand controls pitch/speed/ harmonic layers.


Controls - 
Left hand fingers: style (0=classic, 4=creative)
Right hand X: speed (left=slow, right=fast)  
Right hand Y: pitch (top=high, bottom=low)
Right hand openness: layers (closed=single notes, open=chords)


Flow - 
Live camera > Hand gesture detection > Music generated as MIDI output


Architecture - 
Chopin MIDI library > preprocess_midi.py > data/chopin_sequences.npz
                           |
                      train_model.py > models/chopin_lstm.pt
                           |
main.py (tracking) < MediaPipe > chopin_model.py (LSTM inference) > Mido MIDI output


Open-source Libraries - 
- OpenCV: used for camera and video capture (https://opencv.org/license/)
- MediaPipe Hands: real-time hand detection (https://github.com/google-ai-edge/mediapipe)
- PyTorch: used for the LSTM neural network (https://github.com/pytorch/tutorials/blob/main/LICENSE)
- Pretty midi: MIDI processing for training (https://github.com/craffel/pretty-midi)
- Mido: generating MIDI outputs (https://github.com/mido/mido/)
- NumPy: creating arrays (https://github.com/numpy/numpy)
- *no external models were used or imported


Contributions - 
1. Gesture mappings: tempereature (creativity for left hand) between .65-.95, right hand openness for chord layers, real time chord generation (tried multiple iterations, but settled on root/3rd/5th/7th/9th)
2. Preprocessing: converted original polyphonic MIDIs into a monophonic pitch grid at 125 ms steps, added a dedicated rest token, limited pitches to a safe piano range, and concatenated all pieces into a long training sequence with short rest gaps between them.
3. LSTM: 2 hidden layers with 256 neurons, sequence length of 48, learning rate of 5e-4, trained for 10 epochs
4. Hand gesture definitions: 4 finger counting (thumb ignored for stability), hand openness via wrist to fingertip distance,frame to frame velocity tracking


Public domain Chopin dataset - 
piano-midi.de/chopin_d.htm
kunstderfuge.com/chopin.htm​
midiworld.com/chopin.htm
Categories: Ballades, Etudes, Impromptus, Nocturnes, Preludes, Waltzes


How to run - 
1. create env and install  dependencies:
conda create -n music_ai python=3.9 -y
conda activate music_ai
pip install torch torchvision torchaudio opencv-python mediapipe mido pretty-midi numpy
2. Preprocess MIDIs
python preprocess_midi.py # creates data/chopin_sequences.npz
3. Train LSTM model
python train_model.py # saves models/chopin_lstm.pt
4.Open GarageBand or requivalent > Click on empty project > select "MIDI/Software Instrument"
5. Run demo
python chopin_model.py # opens camera and sends MIDI output; remember Garage Band or equivalent must be open for the model to work 





