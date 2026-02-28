document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const fileName = document.getElementById('file-name');
    const processBtn = document.getElementById('process-btn');
    const verdictBadge = document.getElementById('verdict-badge');
    const reasonsList = document.getElementById('reasons-list');
    const resultCard = document.getElementById('result-card');
    const auditDetails = document.getElementById('audit-details');
    const historyCard = document.getElementById('history-card');
    const historyBody = document.getElementById('history-body');
    const clearHistoryBtn = document.getElementById('clear-history-btn');

    let currentFileData = null;
    let historyLog = [];

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
            addToHistory(file.name, data);
        } catch (error) {
            console.error(error);
        } finally {
            processBtn.disabled = false;
            processBtn.textContent = 'Run Acoustic Audit';
        }
    });

    // Clear history
    clearHistoryBtn.addEventListener('click', () => {
        historyLog = [];
        historyBody.innerHTML = '';
        historyCard.style.display = 'none';
    });

    function addToHistory(filename, data) {
        const metrics = (data.audio_audit && data.audio_audit.metrics) || {};
        const mos_scores = metrics.mos || {};
        const rowNum = historyLog.length + 1;

        historyLog.push({ filename, data });

        const status = data.is_reliable ? 'CLEAN' : 'BAD';
        const score = data.reliability_score !== undefined ? `${data.reliability_score.toFixed(1)}%` : '--';
        const snrVal = metrics.snr !== undefined ? metrics.snr.toFixed(1) : '--';
        const mosVal = mos_scores.ovrl_mos ? mos_scores.ovrl_mos.toFixed(1) : '--';
        const cleaned = data.cleaned_file_url ? '✅ Yes' : '—';
        const statusColor = data.is_reliable ? 'var(--accent-green)' : 'var(--accent-red)';
        const scoreColor = data.reliability_score >= 80 ? 'var(--accent-green)' : data.reliability_score >= 50 ? 'var(--gold-primary)' : 'var(--accent-red)';

        const tr = document.createElement('tr');
        tr.style.borderBottom = '1px solid rgba(255,255,255,0.06)';
        tr.innerHTML = `
            <td style="padding: 10px 12px; color: var(--text-dim);">${rowNum}</td>
            <td style="padding: 10px 12px; font-size: 0.8rem; color: #fff; max-width: 200px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;" title="${filename}">${filename}</td>
            <td style="padding: 10px 12px; font-weight: 800; color: ${statusColor};">${status}</td>
            <td style="padding: 10px 12px; font-weight: 800; color: ${scoreColor};">${score}</td>
            <td style="padding: 10px 12px; color: #ccc;">${snrVal}</td>
            <td style="padding: 10px 12px; color: #ccc;">${mosVal}</td>
            <td style="padding: 10px 12px; color: var(--accent-green);">${cleaned}</td>
        `;

        historyBody.appendChild(tr);
        historyCard.style.display = 'block';
    }

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
            document.getElementById('stat-reliability').textContent = `${data.reliability_score.toFixed(1)}%`;

            // Color code the score
            const relEl = document.getElementById('stat-reliability');
            if (data.reliability_score >= 80) relEl.style.color = 'var(--accent-green)';
            else if (data.reliability_score >= 50) relEl.style.color = 'var(--gold-primary)';
            else relEl.style.color = 'var(--accent-red)';
        }

        // 4.5 Generative Acoustic Summary
        const summaryContainer = document.getElementById('acoustic-summary-container');
        const summaryText = document.getElementById('acoustic-summary-text');

        const metrics = (data.audio_audit && data.audio_audit.metrics) || {};
        const mos_scores = metrics.mos || {};

        if (summaryContainer && summaryText && data.audio_audit) {
            let summarySteps = [];
            const snr = metrics.snr || 0;
            const speech = (metrics.ns_ratio || 0) * 100;
            const silence = 100 - speech;

            if (snr < 15) summarySteps.push(`Significant background noise detected (${snr.toFixed(1)} dB Signal-to-Noise Ratio).`);
            else if (snr < 30) summarySteps.push(`Moderate background noise present (${snr.toFixed(1)} dB Signal-to-Noise Ratio).`);
            else summarySteps.push(`Background Signal-to-Noise Ratio is at highly optimal levels (${snr.toFixed(1)} dB SNR).`);

            if (mos_scores.ovrl_mos) {
                summarySteps.push(`The Microsoft AI Neural Network (DNSMOS) grades the physical shape of the human voice at ${mos_scores.ovrl_mos.toFixed(1)} out of 5.0.`);
            }

            if (silence > 50) summarySteps.push(`The audio file is dominated by dead air and silence (${silence.toFixed(1)}%).`);
            else if (silence > 30) summarySteps.push(`There are some pauses in the recording (${silence.toFixed(1)}% dead air silence).`);
            else summarySteps.push(`Voice Activity Detection (Speech Density) is very thick, making it highly active (${speech.toFixed(1)}% speech).`);

            if (data.is_reliable) {
                summarySteps.push(`Based on this acoustic analysis, the audio file is robust enough to be sent directly to the Deepgram Transcription engine without major errors.`);
            } else {
                if (data.cleaned_file_url) {
                    summarySteps.push(`Because this audio was degraded, it has been automatically cleaned by DeepFilterNet. The cleaned version is now available for playback below.`);
                } else {
                    summarySteps.push(`Because this audio file is highly degraded, it will likely produce semantic hallucinations if transcribed.`);
                }
            }

            summaryText.innerHTML = summarySteps.join(' ');
            summaryContainer.style.display = 'block';
        }

        // 5. Stats & Playback (Audio Focus)
        const snr = metrics.snr;
        const speech = metrics.ns_ratio !== undefined ? metrics.ns_ratio * 100 : undefined;
        const silence = speech !== undefined ? 100 - speech : undefined;
        const clipping = metrics.clipping !== undefined ? metrics.clipping * 100 : undefined;
        const ovrl_mos = mos_scores.ovrl_mos;

        document.getElementById('stat-snr').textContent = snr !== undefined ? `${snr.toFixed(1)}` : '--';
        document.getElementById('stat-speech').textContent = speech !== undefined ? `${speech.toFixed(1)}%` : '--';
        document.getElementById('stat-silence').textContent = silence !== undefined ? `${silence.toFixed(1)}%` : '--';
        document.getElementById('stat-clipping').textContent = clipping !== undefined ? `${clipping.toFixed(1)}%` : '--';
        document.getElementById('stat-mos').textContent = ovrl_mos ? `${ovrl_mos.toFixed(1)} / 5` : '--';

        const audioStatusEl = document.getElementById('stat-audio-status');
        if (data.audio_audit && data.audio_audit.is_noisy) {
            audioStatusEl.innerHTML = '<span style="color: var(--accent-red);">FLAGGED</span>';
        } else {
            audioStatusEl.innerHTML = '<span style="color: var(--accent-green);">CLEAN</span>';
        }

        // 5. Stats & Playback - Always show original, conditionally show cleaned
        const rawPlayer = document.getElementById('raw-player');
        const cleanedPlayer = document.getElementById('cleaned-player');
        const cleanedContainer = document.getElementById('cleaned-audio-container');

        // Always populate the original raw audio
        if (data.raw_file_url) {
            rawPlayer.src = data.raw_file_url;
        }

        // Show cleaned comparison panel only if DeepFilterNet produced a result
        if (data.cleaned_file_url) {
            cleanedPlayer.src = data.cleaned_file_url;
            cleanedContainer.style.display = 'block';
        } else {
            cleanedContainer.style.display = 'none';
        }
    }
});
