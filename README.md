<h2> MP3 Granulator </h2>

This is a granulator synth that takes in an audio file and randomly chooses grains (short sections of the sample of length grain_size) and assembles them into an output the same size as the input audio. 

<h3> grain_size </h3>
This is the size of the choses grain and the size oscilates following a sine wave with a frequency parameter. Shown is a sine wave that has been normalized to a minimum of 20 milliseconds and a maximum of 5000 milliseconds.

<h3> order of choosing grains </h3>
When choosing the sample, the next sample will be randomly choosen from a range 10 * grain_size.  

<h3> what happens next </h3>
Next we calculate a crossfade that slowely softens the previous grain and the current grain to make the transition between each grain more smooth. Then the grain is connected to the previous grain, overlaying by small percent. This continues until the current position index reaches the outputs size (which equals the input audio file).
    
<h3> how to run </h3>
Need libraries pydub, numpy, random, and math. And a
mp3 audio file to granulate. Run the last line to install nonstandard python libraries numpy and pydub
