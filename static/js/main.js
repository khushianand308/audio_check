document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const fileName = document.getElementById('file-name');
    const processBtn = document.getElementById('process-btn');
    const verdictBadge = document.getElementById('verdict-badge');
    const reasonsList = document.getElementById('reasons-list');
    const resultCard = document.getElementById('result-card');
    const auditDetails = document.getElementById('audit-details');
    const cleanedPlayer = document.getElementById('cleaned-player');

    let currentFileData = null;

    // File Selection
    dropZone.addEventListener('click', () => fileInput.click());

    fileInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
            fileName.textContent = file.name;
            processBtn.disabled = false;
        }
    });

    // Handle Upload & Process
    processBtn.addEventListener('click', async () => {
        const file = fileInput.files[0];
        if (!file) return;

        processBtn.disabled = true;
        processBtn.textContent = 'Auditing Acoustic Payload...';

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch('/api/process', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();
            currentFileData = data;

            displayResults(data);
        } catch (error) {
            console.error(error);
        } finally {
            processBtn.disabled = false;
            processBtn.textContent = 'Run Acoustic Audit';
        }
    });

    function displayResults(data) {
        auditDetails.style.display = 'block';

        // 2. Verdict (Explicit Transcript Readiness)
        verdictBadge.style.display = 'block';
        if (data.is_reliable) {
            verdictBadge.textContent = 'APPROVED FOR TRANSCRIPT';
            verdictBadge.className = 'badge badge-clean';
            resultCard.style.borderColor = 'var(--accent-green)';
        } else {
            verdictBadge.textContent = 'DO NOT TRANSCRIBE (NOISY)';
            verdictBadge.className = 'badge badge-bad';
            resultCard.style.borderColor = 'var(--accent-red)';
        }

        // 3. Flags
        reasonsList.innerHTML = '';
        if (data.flags && data.flags.length > 0) {
            data.flags.forEach(flag => {
                const div = document.createElement('div');
                div.style.marginBottom = '0.7rem';
                div.style.padding = '10px';
                div.style.background = 'rgba(255, 76, 76, 0.1)';
                div.style.borderRadius = '6px';
                div.innerHTML = `<span style="color:var(--accent-red); font-weight:800; margin-right:10px;">FLAGGED</span> ${flag}`;
                reasonsList.appendChild(div);
            });
        } else {
            reasonsList.innerHTML = '<div style="padding:10px; background:rgba(46, 204, 113, 0.1); border-radius:6px; color:var(--accent-green);">Optimal signal quality. No semantic hallucinations detected.</div>';
        }

        // 4. Final Reliability Score
        if (data.reliability_score !== undefined) {
            document.getElementById('reliability-container').style.display = 'block';
            document.getElementById('stat-reliability').textContent = `${data.reliability_score}%`;

            // Color code the score
            const relEl = document.getElementById('stat-reliability');
            if (data.reliability_score >= 80) relEl.style.color = 'var(--accent-green)';
            else if (data.reliability_score >= 50) relEl.style.color = 'var(--gold-primary)';
            else relEl.style.color = 'var(--accent-red)';
        }

        // 4.5 Generative Acoustic Summary
        const summaryContainer = document.getElementById('acoustic-summary-container');
        const summaryText = document.getElementById('acoustic-summary-text');
        if (summaryContainer && summaryText && data.audio_audit) {
            let summarySteps = [];
            const snr = data.audio_audit.snr_db;
            const silence = data.audio_audit.silence_ratio * 100;
            const speech = data.audio_audit.non_silence_ratio * 100;

            if (snr < 15) summarySteps.push(`Significant background noise detected (${snr.toFixed(1)} dB Signal-to-Noise Ratio).`);
            else if (snr < 30) summarySteps.push(`Moderate background noise present (${snr.toFixed(1)} dB Signal-to-Noise Ratio).`);
            else summarySteps.push(`Background Signal-to-Noise Ratio is at highly optimal levels (${snr.toFixed(1)} dB SNR).`);

            if (data.audio_audit.ovrl_mos) {
                summarySteps.push(`The Microsoft AI Neural Network (DNSMOS) grades the physical shape of the human voice at ${data.audio_audit.ovrl_mos.toFixed(1)} out of 5.0.`);
            }

            if (silence > 50) summarySteps.push(`The audio file is dominated by dead air and silence (${silence.toFixed(1)}%).`);
            else if (silence > 30) summarySteps.push(`There are some pauses in the recording (${silence.toFixed(1)}% dead air silence).`);
            else summarySteps.push(`Voice Activity Detection (Speech Density) is very thick, making it highly active (${speech.toFixed(1)}% speech).`);

            if (data.is_reliable) {
                summarySteps.push(`Based on this acoustic analysis, the audio file is robust enough to be sent directly to the Deepgram Transcription engine without major errors.`);
            } else {
                summarySteps.push(`Because this audio file is highly degraded, it will likely produce semantic hallucinations if transcribed. It must be cleaned through the DeepFilterNet suppression pipeline first.`);
            }

            summaryText.innerHTML = summarySteps.join(' ');
            summaryContainer.style.display = 'block';
        }


        // 5. Stats & Playback (Audio Focus)
        document.getElementById('stat-snr').textContent = data.audio_audit.snr_db ? `${data.audio_audit.snr_db.toFixed(1)}` : '--';
        document.getElementById('stat-speech').textContent = data.audio_audit.non_silence_ratio ? `${(data.audio_audit.non_silence_ratio * 100).toFixed(1)}%` : '--';
        document.getElementById('stat-silence').textContent = data.audio_audit.silence_ratio ? `${(data.audio_audit.silence_ratio * 100).toFixed(1)}%` : '--';
        document.getElementById('stat-clipping').textContent = data.audio_audit.clipping_ratio !== undefined ? `${(data.audio_audit.clipping_ratio * 100).toFixed(1)}%` : '--';
        document.getElementById('stat-mos').textContent = data.audio_audit.ovrl_mos ? `${data.audio_audit.ovrl_mos.toFixed(1)} / 5` : '--';

        const audioStatusEl = document.getElementById('stat-audio-status');
        if (data.audio_audit && data.audio_audit.is_noisy) {
            audioStatusEl.innerHTML = '<span style="color: var(--accent-red);">FLAGGED</span>';
        } else {
            audioStatusEl.innerHTML = '<span style="color: var(--accent-green);">CLEAN</span>';
        }

        if (data.cleaned_file_url) {
            document.getElementById('audio-preview-label').textContent = "CLEANED AUDIO PREVIEW (AI ENHANCED)";
            cleanedPlayer.src = data.cleaned_file_url;
            cleanedPlayer.style.display = 'block';
        } else {
            document.getElementById('audio-preview-label').textContent = "RAW AUDIO PREVIEW (CLEAN SIGNAL)";
            // Use the raw file URL served directly from the backend
            if (data.raw_file_url) {
                cleanedPlayer.src = data.raw_file_url;
                cleanedPlayer.style.display = 'block';
            }
        }
    }
});
