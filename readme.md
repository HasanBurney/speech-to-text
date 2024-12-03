<h2><strong>Prerequisites</strong></h2>
<ol>
  <li><strong>Hardware:</strong> NVIDIA GPU with CUDA support.(Else set the whisper model using the following line => 
  <br><code>whisper_model = WhisperModel("deepdml/faster-whisper-large-v3-turbo-ct2", device="cpu", compute_type="float16")
)</code></li>
  <li><strong>Python:</strong> Version 3.8 or later.</li>
  <li><strong>Ollama:</strong> Installed and running locally. Download from <a href="https://ollama.ai">ollama.ai</a>.</li>
  <li><strong>CUDA Toolkit:</strong> Installed from <a href="https://developer.nvidia.com/cuda-toolkit">NVIDIA's CUDA Toolkit page</a>.</li>
</ol>

<hr>

<h2><strong>Setup</strong></h2>

<h3><strong>Step 1: Create a Virtual Environment</strong></h3>
<pre><code>python -m venv env
./env/Scripts/activate
</code></pre>

<h3><strong>Step 2: Install Dependencies</strong></h3>
<pre><code>pip install -r requirements.txt
</code></pre>

<h3><strong>Step 3: Start Ollama Service</strong></h3>
<pre><code>ollama pull llama3.2
</code></pre>
<pre><code>ollama serve
</code></pre>

<h3><strong>Step 4: Run the Script</strong></h3>
<pre><code>python transcriber.py
</code></pre>

<hr>

<h3><strong>Notes</strong></h3>
<ul>
  <li>Ensure the <code>slots.txt</code> file is in the same directory to track appointment slots.</li>
  <li>Ensure your microphone is accessible</li>
  <li>Confirm the Ollama API is accessible at <a href="http://localhost:11434">http://localhost:11434</a>.</li>
</ul>
