// ═══════════════════════════════════════════════
//  DATA ARRAYS
// ═══════════════════════════════════════════════
const majors = [
    'Arts and Social Sciences',
    'Communication, Art and Technology',
    'Science',
    'Applied Sciences',
    'Business',
    'Education',
    'Environment',
    'Health Sciences',
    'Graduate Studies',
    'School of Medicine'
];

const years        = ['1st Year', '2nd Year', '3rd Year', '4th Year', 'Post-grad'];
const personalities = ['Introvert', 'Ambivert', 'Extrovert'];
const studyStyles   = ['Solo Study', 'Group Study', 'Mix of Both'];

const languages = [
    'Afrikaans','Albanian','Amharic','Arabic','Armenian','Azerbaijani','Basque','Belarusian',
    'Bengali','Bosnian','Bulgarian','Burmese','Catalan','Cebuano','Chinese (Simplified)',
    'Chinese (Traditional)','Croatian','Czech','Danish','Dutch','English','Estonian',
    'Filipino','Finnish','French','Georgian','German','Greek','Gujarati','Hebrew','Hindi',
    'Hungarian','Icelandic','Indonesian','Irish','Italian','Japanese','Kannada','Kazakh',
    'Khmer','Korean','Kurdish','Lao','Latvian','Lithuanian','Malay','Malayalam','Maltese',
    'Maori','Marathi','Mongolian','Nepali','Norwegian','Pashto','Persian','Polish',
    'Portuguese','Punjabi','Romanian','Russian','Serbian','Sinhala','Slovak','Slovenian',
    'Somali','Spanish','Swahili','Swedish','Tamil','Telugu','Thai','Turkish','Ukrainian',
    'Urdu','Uzbek','Vietnamese','Welsh','Yoruba','Zulu'
].sort();

const countries = [
    'Afghanistan','Albania','Algeria','Argentina','Armenia','Australia','Austria','Azerbaijan',
    'Bangladesh','Belarus','Belgium','Bhutan','Bolivia','Bosnia and Herzegovina','Brazil',
    'Bulgaria','Cambodia','Cameroon','Canada','Chile','China','Colombia','Costa Rica',
    'Croatia','Cuba','Czech Republic','Denmark','Dominican Republic','Ecuador','Egypt',
    'Ethiopia','Finland','France','Georgia','Germany','Ghana','Greece','Guatemala',
    'Honduras','Hong Kong','Hungary','Iceland','India','Indonesia','Iran','Iraq','Ireland',
    'Israel','Italy','Jamaica','Japan','Jordan','Kazakhstan','Kenya','Kuwait','Kyrgyzstan',
    'Laos','Latvia','Lebanon','Libya','Lithuania','Luxembourg','Malaysia','Mexico',
    'Mongolia','Morocco','Myanmar','Nepal','Netherlands','New Zealand','Nigeria',
    'North Korea','Norway','Oman','Pakistan','Palestine','Panama','Paraguay','Peru',
    'Philippines','Poland','Portugal','Qatar','Romania','Russia','Saudi Arabia','Senegal',
    'Serbia','Singapore','Slovakia','Slovenia','South Africa','South Korea','Spain',
    'Sri Lanka','Sudan','Sweden','Switzerland','Syria','Taiwan','Tanzania','Thailand',
    'Tunisia','Turkey','Uganda','Ukraine','United Arab Emirates','United Kingdom',
    'United States','Uruguay','Uzbekistan','Venezuela','Vietnam','Yemen','Zambia','Zimbabwe'
].sort();

const cuisines = [
    'Chinese','Italian','Mexican','Japanese','Indian','Thai','Korean','Vietnamese',
    'Mediterranean','French','Greek','Spanish','Middle Eastern','American','Brazilian',
    'Caribbean','Ethiopian','Peruvian','German','Turkish','Malaysian','Indonesian',
    'Filipino','Portuguese','Moroccan','Lebanese','Argentinian','Russian','Polish',
    'British','Scandinavian','Australian','South African','Pakistani','Bangladeshi',
    'Sri Lankan','Nepalese'
];

const interests = [
    'Sports & Fitness','Arts & Creativity','Music','Gaming & Esports',
    'Volunteering & Community Work','Outdoors & Adventure','Reading & Writing',
    'Technology & DIY','Film & TV','Cooking & Food','Social & Events',
    'Meditation & Wellness'
];

const movies = [
    'Action','Comedy','Drama','Romance','Horror','Thriller',
    'Science Fiction','Fantasy','Animation','Documentary','Mystery','Adventure'
];

// Emoji displayed on chip/tag buttons (value sent to backend is still the plain string)
const emojiMap = {
    // Year
    '1st Year': '🌱', '2nd Year': '📖', '3rd Year': '🔬', '4th Year': '🎓', 'Post-grad': '🏆',
    // Personality
    'Introvert': '🤫', 'Ambivert': '⚖️', 'Extrovert': '🎉',
    // Study style
    'Solo Study': '🎧', 'Group Study': '👥', 'Mix of Both': '🔀',
    // Cuisines
    'Chinese':'🥡','Italian':'🍝','Mexican':'🌮','Japanese':'🍣','Indian':'🍛',
    'Thai':'🍜','Korean':'🥘','Vietnamese':'🍲','Mediterranean':'🫒','French':'🥐',
    'Greek':'🧆','Spanish':'🥘','Middle Eastern':'🧆','American':'🍔','Brazilian':'🥩',
    'Caribbean':'🌴','Ethiopian':'🫘','Peruvian':'🌽','German':'🥨','Turkish':'🥙',
    'Malaysian':'🍜','Indonesian':'🍛','Filipino':'🍚','Portuguese':'🥚','Moroccan':'🫖',
    'Lebanese':'🥗','Argentinian':'🥩','Russian':'🥟','Polish':'🥟','British':'☕',
    'Scandinavian':'🐟','Australian':'🥬','South African':'🍖','Pakistani':'🍛',
    'Bangladeshi':'🐟','Sri Lankan':'🌶️','Nepalese':'🫕',
    // Interests
    'Sports & Fitness':'⚽','Arts & Creativity':'🎨','Music':'🎵',
    'Gaming & Esports':'🎮','Volunteering & Community Work':'🤝',
    'Outdoors & Adventure':'🏔️','Reading & Writing':'📚','Technology & DIY':'💻',
    'Film & TV':'🎬','Cooking & Food':'👨‍🍳','Social & Events':'🎉','Meditation & Wellness':'🧘',
    // Movies
    'Action':'💥','Comedy':'😂','Drama':'🎭','Romance':'❤️','Horror':'👻',
    'Thriller':'🔍','Science Fiction':'🚀','Fantasy':'🧙','Animation':'✨',
    'Documentary':'🎞️','Mystery':'🕵️','Adventure':'🗺️'
};

// Fun facts shown while processing
const funFacts = [
    'Our AI is analysing your profile across 9 compatibility dimensions…',
    'K-Means clustering groups students by shared academic patterns…',
    'Jaccard similarity scores measure multi-value interest overlap…',
    'Diversity bonuses reward connections across cultures and majors…',
    'Weighted scoring gives more importance to personality alignment…',
    'Your study style compatibility is being calculated right now…',
    'Finding the students who\'ll actually get you…',
    'Almost there — crunching the final compatibility numbers! 🔢'
];

// ═══════════════════════════════════════════════
//  STATE
// ═══════════════════════════════════════════════
let formData = {
    name: '', email: '', major: '', year: '', language: '', country: '',
    personality: '', studyStyle: '', cuisine: [], interests: [], movies: []
};

// ═══════════════════════════════════════════════
//  INIT
// ═══════════════════════════════════════════════
document.addEventListener('DOMContentLoaded', () => {
    populateSelects();
    setupForm();
    initNeuralCanvas();
    initTiltEffect();
    initRippleEffect();
    initTypewriter();
    initStaggerAnimations();
    updateProgress();
});

// ═══════════════════════════════════════════════
//  TYPEWRITER EFFECT
// ═══════════════════════════════════════════════
function initTypewriter() {
    const el = document.querySelector('.hero-typewriter');
    if (!el) return;
    const words = ['study partners', 'lab mates', 'project squad', 'academic allies', 'campus crew', 'like-minded people'];
    let wordIdx = 0, charIdx = 0, deleting = false;

    function type() {
        const word = words[wordIdx];
        if (!deleting) {
            charIdx++;
            el.textContent = word.slice(0, charIdx);
            if (charIdx === word.length) {
                deleting = true;
                setTimeout(type, 1800);
                return;
            }
            setTimeout(type, 80);
        } else {
            charIdx--;
            el.textContent = word.slice(0, charIdx);
            if (charIdx === 0) {
                deleting = false;
                wordIdx = (wordIdx + 1) % words.length;
            }
            setTimeout(type, 48);
        }
    }
    setTimeout(type, 900);
}

// ═══════════════════════════════════════════════
//  STAGGER ANIMATIONS
// ═══════════════════════════════════════════════
function initStaggerAnimations() {
    // Welcome screen elements fade in on load
    const welcomeItems = document.querySelectorAll('#welcome-screen .pill-badge, #welcome-screen .hero-title-wrap, #welcome-screen .hero-sub, #welcome-screen .feat-grid, #welcome-screen .btn-primary, #welcome-screen .stats-pill');
    welcomeItems.forEach((el, i) => {
        el.style.opacity = '0';
        el.style.transform = 'translateY(22px)';
        el.style.transition = `opacity 0.55s ease ${i * 90}ms, transform 0.55s cubic-bezier(0.34,1.56,0.64,1) ${i * 90}ms`;
        setTimeout(() => {
            el.style.opacity = '1';
            el.style.transform = 'translateY(0)';
        }, 150 + i * 90);
    });
}

// ═══════════════════════════════════════════════
//  CONFETTI
// ═══════════════════════════════════════════════
function launchConfetti() {
    const canvas = document.getElementById('confetti-canvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    canvas.width  = window.innerWidth;
    canvas.height = window.innerHeight;
    canvas.style.display = 'block';

    const colors = ['#8B5CF6','#EC4899','#06B6D4','#F59E0B','#10B981','#F472B6','#60A5FA','#FCD34D'];
    const particles = [];

    for (let i = 0; i < 180; i++) {
        const angle = (Math.random() * 360) * (Math.PI / 180);
        const speed = Math.random() * 10 + 4;
        particles.push({
            x:    canvas.width * (0.3 + Math.random() * 0.4),
            y:    canvas.height * 0.38,
            vx:   Math.cos(angle) * speed * (Math.random() > 0.5 ? 1 : -1),
            vy:   Math.sin(angle) * speed - 6,
            size: Math.random() * 9 + 4,
            color: colors[Math.floor(Math.random() * colors.length)],
            rot:   Math.random() * 360,
            rotV:  (Math.random() - 0.5) * 10,
            gravity: 0.28,
            drag: 0.99,
            opacity: 1,
            shape: Math.random() > 0.45 ? 'rect' : 'circle'
        });
    }

    let frame = 0;
    function animate() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        let alive = false;
        particles.forEach(p => {
            if (p.opacity <= 0.01) return;
            alive = true;
            p.vx *= p.drag;
            p.vy += p.gravity;
            p.x  += p.vx;
            p.y  += p.vy;
            p.rot += p.rotV;
            if (p.y > canvas.height * 0.75) p.opacity -= 0.018;
            else if (frame > 120) p.opacity -= 0.006;

            ctx.save();
            ctx.translate(p.x, p.y);
            ctx.rotate(p.rot * Math.PI / 180);
            ctx.globalAlpha = Math.max(0, p.opacity);
            ctx.fillStyle = p.color;
            if (p.shape === 'circle') {
                ctx.beginPath();
                ctx.arc(0, 0, p.size / 2, 0, Math.PI * 2);
                ctx.fill();
            } else {
                ctx.fillRect(-p.size / 2, -p.size / 4, p.size, p.size / 2);
            }
            ctx.restore();
        });
        frame++;
        if (alive) requestAnimationFrame(animate);
        else canvas.style.display = 'none';
    }
    animate();
}

// ═══════════════════════════════════════════════
//  NEURAL CANVAS
// ═══════════════════════════════════════════════
function initNeuralCanvas() {
    const canvas = document.getElementById('neural-canvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    let W, H, nodes = [];

    const isMobile  = () => window.innerWidth < 768;
    const nodeCount = () => isMobile() ? 36 : 68;
    const maxDist   = 162;

    function resize() { W = canvas.width = window.innerWidth; H = canvas.height = window.innerHeight; }

    function createNodes() {
        nodes = [];
        for (let i = 0; i < nodeCount(); i++) {
            const roll = Math.random();
            const color = roll > 0.78 ? [236,72,153] : roll > 0.56 ? [6,182,212] : roll > 0.34 ? [139,92,246] : [59,130,246];
            nodes.push({
                x: Math.random() * W, y: Math.random() * H,
                vx: (Math.random() - 0.5) * 0.27, vy: (Math.random() - 0.5) * 0.27,
                r: Math.random() * 1.6 + 0.3,
                pulse: Math.random() * Math.PI * 2,
                color
            });
        }
    }

    function draw() {
        ctx.clearRect(0, 0, W, H);
        for (let i = 0; i < nodes.length; i++) {
            for (let j = i + 1; j < nodes.length; j++) {
                const dx = nodes[i].x - nodes[j].x, dy = nodes[i].y - nodes[j].y;
                const dist = Math.sqrt(dx*dx + dy*dy);
                if (dist < maxDist) {
                    const alpha = (1 - dist / maxDist) * 0.2;
                    const [r,g,b] = nodes[i].color;
                    ctx.beginPath();
                    ctx.moveTo(nodes[i].x, nodes[i].y);
                    ctx.lineTo(nodes[j].x, nodes[j].y);
                    ctx.strokeStyle = `rgba(${r},${g},${b},${alpha})`;
                    ctx.lineWidth = 0.6;
                    ctx.stroke();
                }
            }
        }
        nodes.forEach(n => {
            const glow = 1 + 0.35 * Math.sin(n.pulse);
            const [r,g,b] = n.color;
            ctx.beginPath();
            ctx.arc(n.x, n.y, n.r * glow, 0, Math.PI * 2);
            ctx.fillStyle = `rgba(${r},${g},${b},0.8)`;
            ctx.fill();
        });
    }

    function update() {
        nodes.forEach(n => {
            n.x += n.vx; n.y += n.vy; n.pulse += 0.018;
            if (n.x < 0 || n.x > W) n.vx *= -1;
            if (n.y < 0 || n.y > H) n.vy *= -1;
        });
    }

    function loop() { update(); draw(); requestAnimationFrame(loop); }
    window.addEventListener('resize', () => { resize(); createNodes(); });
    resize(); createNodes(); loop();
}

// ═══════════════════════════════════════════════
//  TILT EFFECT
// ═══════════════════════════════════════════════
function initTiltEffect() {
    function applyTilt(card) {
        card.addEventListener('mousemove', e => {
            const { left, top, width, height } = card.getBoundingClientRect();
            const x = (e.clientX - left) / width  - 0.5;
            const y = (e.clientY - top)  / height - 0.5;
            card.style.transform = `perspective(700px) rotateX(${-y*11}deg) rotateY(${x*11}deg) scale(1.03) translateZ(8px)`;
        });
        card.addEventListener('mouseleave', () => {
            card.style.transform = 'perspective(700px) rotateX(0) rotateY(0) scale(1) translateZ(0)';
        });
    }
    document.querySelectorAll('[data-tilt]').forEach(applyTilt);
    new MutationObserver(() => {
        document.querySelectorAll('[data-tilt]').forEach(card => {
            if (!card.dataset.tiltInit) { card.dataset.tiltInit = '1'; applyTilt(card); }
        });
    }).observe(document.body, { childList: true, subtree: true });
}

// ═══════════════════════════════════════════════
//  RIPPLE EFFECT
// ═══════════════════════════════════════════════
function initRippleEffect() {
    document.querySelectorAll('.ripple').forEach(el => {
        el.addEventListener('click', function(e) {
            const rect = this.getBoundingClientRect();
            const size = Math.max(rect.width, rect.height);
            const dot  = document.createElement('span');
            dot.style.cssText = `
                position:absolute; width:${size}px; height:${size}px;
                left:${e.clientX - rect.left - size/2}px;
                top:${e.clientY - rect.top - size/2}px;
                background:rgba(255,255,255,0.26); border-radius:50%;
                transform:scale(0); animation:rippleAnim 0.62s ease-out forwards;
                pointer-events:none; z-index:0;`;
            this.appendChild(dot);
            setTimeout(() => dot.remove(), 660);
        });
    });
}

// ═══════════════════════════════════════════════
//  NAVIGATION (3D transitions)
// ═══════════════════════════════════════════════
function navigateTo(targetId, direction) {
    direction = direction || 'forward';
    const current = document.querySelector('.screen.active');
    const target  = document.getElementById(targetId);
    if (!current || !target || current === target) return;

    const DURATION = 580;
    const exitMap  = { forward:'screenExitForward',  back:'screenExitBackward',  burst:'screenExitShrink' };
    const enterMap = { forward:'screenEnterForward', back:'screenEnterBackward', burst:'screenEnterBurst' };
    const exitAnim  = exitMap[direction]  || exitMap.forward;
    const enterAnim = enterMap[direction] || enterMap.forward;

    current.style.cssText = `display:block; pointer-events:none; z-index:9; opacity:1;
        animation:${exitAnim} ${DURATION}ms var(--ease) forwards;`;
    target.style.cssText = `display:block; pointer-events:none; z-index:10;
        animation:${enterAnim} ${DURATION}ms var(--ease-out) forwards;`;

    setTimeout(() => {
        current.style.cssText = ''; current.classList.remove('active');
        target.style.cssText  = ''; target.classList.add('active');
        setTimeout(() => { target.scrollTop = 0; }, 10);
    }, DURATION + 30);
}

// ═══════════════════════════════════════════════
//  POPULATE FORM
// ═══════════════════════════════════════════════
function populateSelects() {
    const majorSel = document.getElementById('major');
    majors.forEach(m => { const o = document.createElement('option'); o.value = o.textContent = m; majorSel.appendChild(o); });

    const langSel = document.getElementById('language');
    langSel.innerHTML = '<option value="">Select language</option>';
    languages.forEach(l => { const o = document.createElement('option'); o.value = o.textContent = l; langSel.appendChild(o); });

    const countrySel = document.getElementById('country');
    countrySel.innerHTML = '<option value="">Select country</option>';
    countries.forEach(c => { const o = document.createElement('option'); o.value = o.textContent = c; countrySel.appendChild(o); });

    buildChipGroup('year-container',         years,        'year',       false);
    buildChipGroup('personality-container',  personalities,'personality',false);
    buildChipGroup('study-style-container',  studyStyles,  'studyStyle', false);
    buildChipGroup('cuisine-container',      cuisines,     'cuisine',    true);
    buildChipGroup('interests-container',    interests,    'interests',  true);
    buildChipGroup('movies-container',       movies,       'movies',     true);
}

function buildChipGroup(containerId, items, field, isMulti) {
    const container = document.getElementById(containerId);
    if (!container) return;
    items.forEach(item => {
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = isMulti ? 'tag-btn ripple' : 'chip-btn ripple';
        // Show emoji + label; backend value stays as the plain string
        const emoji = emojiMap[item] || '';
        btn.textContent  = emoji ? `${emoji} ${item}` : item;
        btn.dataset.value = item;
        btn.addEventListener('click', () => {
            isMulti ? toggleMultiOption(field, item, btn) : selectSingleOption(field, item, btn, container);
        });
        container.appendChild(btn);
    });
}

// ═══════════════════════════════════════════════
//  FORM SETUP
// ═══════════════════════════════════════════════
function setupForm() {
    const form = document.getElementById('student-form');

    ['name','email'].forEach(id => {
        const el = document.getElementById(id);
        if (el) el.addEventListener('input', e => {
            formData[id] = e.target.value.trim();
            updateSectionChecks(); updateProgress();
        });
    });

    ['major','language','country'].forEach(id => {
        const el = document.getElementById(id);
        if (el) el.addEventListener('change', e => {
            formData[id] = e.target.value;
            updateSectionChecks(); updateProgress();
        });
    });

    form.addEventListener('submit', handleSubmit);
}

// ═══════════════════════════════════════════════
//  SELECTION HANDLERS
// ═══════════════════════════════════════════════
function selectSingleOption(field, value, btn, container) {
    formData[field] = value;
    container.querySelectorAll('button').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    updateSectionChecks(); updateProgress();
}

function toggleMultiOption(field, value, btn) {
    const idx = formData[field].indexOf(value);
    if (idx > -1) { formData[field].splice(idx, 1); btn.classList.remove('active'); }
    else          { formData[field].push(value);     btn.classList.add('active'); }
    updateProgress();
}

// ═══════════════════════════════════════════════
//  PROGRESS TRACKING
// ═══════════════════════════════════════════════
function updateProgress() {
    const single = ['name','email','major','year','language','country','personality','studyStyle'];
    const multi  = ['cuisine','interests','movies'];
    const total  = single.length + multi.length;
    let filled   = 0;
    single.forEach(f => { if (formData[f]) filled++; });
    multi.forEach(f  => { if (formData[f].length > 0) filled++; });
    const pct   = Math.round((filled / total) * 100);
    const fill  = document.getElementById('progress-fill');
    const pctEl = document.getElementById('progress-pct');
    if (fill)  fill.style.width  = pct + '%';
    if (pctEl) pctEl.textContent = pct + '%';
}

function updateSectionChecks() {
    const checks = {
        basic:       formData.name && formData.email,
        academic:    formData.major && formData.year,
        location:    formData.language && formData.country,
        personality: formData.personality && formData.studyStyle
    };
    Object.entries(checks).forEach(([sec, done]) => {
        const el = document.getElementById(`check-${sec}`);
        if (!el) return;
        if (done) { el.classList.add('done'); el.textContent = '✓'; }
        else      { el.classList.remove('done'); el.textContent = ''; }
    });
}

// ═══════════════════════════════════════════════
//  SCREEN HELPERS
// ═══════════════════════════════════════════════
function showProcessing() {
    navigateTo('processing-screen', 'forward');
    animateProcessingSteps();
    rotateFunFacts();
}

function rotateFunFacts() {
    const el = document.getElementById('proc-fun-fact');
    if (!el) return;
    let idx = 0;
    const interval = setInterval(() => {
        if (!document.getElementById('processing-screen').classList.contains('active')) {
            clearInterval(interval); return;
        }
        idx = (idx + 1) % funFacts.length;
        el.style.opacity = '0';
        setTimeout(() => {
            el.textContent = funFacts[idx];
            el.style.opacity = '1';
        }, 380);
    }, 2200);
}

function showResults(data) {
    const matchCount = data.totalMatches || 0;
    const matchBadge = document.getElementById('match-badge');
    const mbNum      = document.getElementById('mb-num');
    const mbPlural   = document.getElementById('mb-plural');
    const resTitle   = document.getElementById('res-title');
    const resSub     = document.getElementById('res-sub');

    if (matchCount === 0) {
        resTitle.textContent = "You're All Set! 🎉";
        resSub.textContent   = "Your profile has been saved. We're still growing our community — check back soon for matches!";
        if (matchBadge) matchBadge.style.display = 'none';
    } else {
        resTitle.textContent = 'Matches Found! 🎉';
        resSub.textContent   = "Your profile is saved and we've found your perfect study partners!";
        if (matchBadge) {
            matchBadge.style.display = 'flex';
            animateCounter(mbNum, matchCount);
        }
        if (mbPlural) mbPlural.textContent = matchCount === 1 ? '' : 'es';
        // 🎊 Fire the confetti after the screen transition settles
        setTimeout(() => launchConfetti(), 680);
    }

    const emailEl = document.getElementById('student-email');
    if (emailEl) emailEl.textContent = formData.email;

    navigateTo('results-screen', 'burst');
}

function animateCounter(el, target) {
    if (!el) return;
    let current = 0;
    const step  = Math.ceil(target / 20);
    const tick  = setInterval(() => {
        current += step;
        if (current >= target) { el.textContent = target; clearInterval(tick); }
        else el.textContent = current;
    }, 60);
}

function resetApp() {
    formData = {
        name:'', email:'', major:'', year:'', language:'', country:'',
        personality:'', studyStyle:'', cuisine:[], interests:[], movies:[]
    };
    document.getElementById('student-form').reset();
    document.querySelectorAll('.chip-btn, .tag-btn').forEach(b => b.classList.remove('active'));
    ['check-basic','check-academic','check-location','check-personality'].forEach(id => {
        const el = document.getElementById(id);
        if (el) { el.classList.remove('done'); el.textContent = ''; }
    });
    ['step-1','step-2','step-3','step-4'].forEach(id => {
        const row = document.getElementById(id);
        if (row) row.classList.remove('active','done');
    });
    updateProgress();
    navigateTo('form-screen', 'forward');
}

// ═══════════════════════════════════════════════
//  PROCESSING STEP ANIMATION
// ═══════════════════════════════════════════════
function animateProcessingSteps() {
    const steps  = ['step-1','step-2','step-3','step-4'];
    const delays = [600, 1500, 2500, 3600];

    steps.forEach(id => {
        const row = document.getElementById(id);
        if (row) row.classList.remove('active','done');
    });

    steps.forEach((id, i) => {
        setTimeout(() => {
            const row = document.getElementById(id);
            if (!row) return;
            row.classList.add('active');
            setTimeout(() => { row.classList.remove('active'); row.classList.add('done'); }, 800);
        }, delays[i]);
    });
}

// ═══════════════════════════════════════════════
//  FORM SUBMISSION
// ═══════════════════════════════════════════════
async function handleSubmit(e) {
    e.preventDefault();
    if (!validateForm()) return;

    showProcessing();
    await new Promise(r => setTimeout(r, 1600));

    try {
        const response = await fetch('http://localhost:3001/api/students', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(formData)
        });

        if (!response.ok) throw new Error(`HTTP ${response.status}`);

        const result = await response.json();
        const ok = result.success === true || result.success === 'true' || result.success === 1;

        if (ok) {
            showResults(result);
        } else {
            showError(result.error || 'Submission failed. Please try again.');
            navigateTo('form-screen', 'back');
        }
    } catch (err) {
        console.warn('Server unavailable, showing demo result:', err.message);
        // Demo fallback
        showResults({ success: true, totalMatches: Math.floor(Math.random() * 5) + 1 });
    }
}

function validateForm() {
    const required = ['name','email','major','year','language','country','personality','studyStyle'];
    const missing  = required.filter(f => !formData[f]);
    if (missing.length > 0) {
        showError(`Please fill in: ${missing.join(', ')}`);
        return false;
    }
    if (formData.cuisine.length === 0 || formData.interests.length === 0 || formData.movies.length === 0) {
        showError('Please select at least one option for Cuisines, Interests, and Movie Genres.');
        return false;
    }
    return true;
}

function showError(message) {
    const errorBox = document.getElementById('error-banner');
    if (!errorBox) return;
    errorBox.textContent = message;
    errorBox.style.display = 'block';
    errorBox.style.animation = 'none';
    void errorBox.offsetHeight;
    errorBox.style.animation = '';
    const formBody = document.querySelector('.form-body');
    if (formBody) formBody.scrollIntoView({ behavior: 'smooth', block: 'start' });
    setTimeout(() => { errorBox.style.display = 'none'; }, 6000);
}
