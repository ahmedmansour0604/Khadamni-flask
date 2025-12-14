function toArray(str) {
    return str.split(",").map(Number);
}

function toggleSidebar() {
    const sidebar = document.querySelector('.sidebar');
    const mainContent = document.querySelector('.main-content');
    
    sidebar.classList.toggle('collapsed');
    mainContent.classList.toggle('expanded');
}

function predictClass() {
    const result = document.getElementById('class_result');
    result.innerHTML = '<div class="job-item"><strong>✅ Offre Légitime</strong><p>Probabilité: 92%</p></div>';
}

function recommendLocation() {
    const result = document.getElementById('cluster_result');
    result.innerHTML = '<div class="job-item"><strong>Cluster: Île-de-France</strong><p>150 offres trouvées</p></div>';
}

function predictSalary() {
    const experience = document.getElementById('experience').value;
    const result = document.getElementById('reg_result');
    const estimatedSalary = 30000 + (parseInt(experience) * 2500);
    result.innerHTML = `<div class="job-item"><strong>💰 Salaire Estimé</strong><p>${estimatedSalary.toLocaleString('fr-FR')} €</p></div>`;
}

function recommendJobs() {
    const result = document.getElementById('reco_result');
    result.innerHTML = '<div class="job-item"><strong>Data Scientist - Tech Corp</strong><p>Paris - 55,000 €</p></div>';
}

// Sample jobs data (replace by API later)
const JOBS = [
    { id:1, title:'Data Scientist', company:'Tech Corp', location:'Paris', type:'Full-time', salary:55000, tags:['Python','ML','SQL'], img:'images/formation8DATA_science.jpg', desc:'Responsable des modèles ML, collecte et pipeline.' },
    { id:2, title:'Machine Learning Engineer', company:'HealthAI', location:'Remote', type:'Remote', salary:65000, tags:['TensorFlow','Docker'], img:'images/Must-Have-Machine-Learning-Engineer-Skills.webp', desc:'Déploiement de modèles pour produits santé.' },
    { id:3, title:'Data Analyst', company:'Finance Lab', location:'Lyon', type:'Part-time', salary:42000, tags:['SQL','PowerBI'], img:'images/images.jpg', desc:'Analyses financières et reporting.' },
    { id:4, title:'AI Product Manager', company:'Startup X', location:'Marseille', type:'CDI', salary:70000, tags:['Product','ML'], img:'images/AI-Product-Management.jpg', desc:'Pilotage produit IA.' },
    { id:5, title:'Junior Data Scientist', company:'DataStart', location:'Paris', type:'Full-time', salary:38000, tags:['Python','Pandas'], img:'images/maxresdefault.jpg', desc:'Apprentissage et support data.' },
    { id:6, title:'Data Management Specialist', company:'DeepLab', location:'Remote', type:'Remote', salary:90000, tags:['Research','PyTorch'], img:'images/screenshot.png', desc:'Recherche avancée ML.' },
    { id:7, title:'BI Developer', company:'InsightCo', location:'Lyon', type:'Full-time', salary:48000, tags:['SQL','ETL'], img:'images/bi-developer-datascientest-1024x512.png', desc:'Conception dashboards.' },
    { id: 8,title: 'Software Engineer',company: 'Innovatech',location: 'Remote', type: 'Full-time',salary: 60000,tags: ['JavaScript', 'React', 'Node.js'],  img: 'images/how-to-be-a-software-engineer.jpg', desc: 'Développement d’applications web modernes, conception backend et intégration API.'
}

];

const PAGE_SIZE = 6;
let currentPage = 1;
let SHOW_ALL = false;

// --- Job persistence helpers (recruiter posts saved in localStorage) ---
function getNextJobId() {
    return JOBS.reduce((m,j)=>Math.max(m, j.id || 0), 0) + 1;
}

function loadSavedJobs() {
    try {
        const saved = JSON.parse(localStorage.getItem('kh_jobs') || '[]');
        if (!Array.isArray(saved) || saved.length === 0) return;
        // ensure each saved job has an id and the same structure
        const maxId = JOBS.reduce((m,j)=>Math.max(m, j.id || 0), 0);
        saved.forEach((s, i) => {
            if (!s.id) s.id = maxId + i + 1;
            // insert saved jobs at the front so they're shown first
            JOBS.unshift(s);
        });
    } catch (e) { console.warn('Failed to load saved jobs', e); }
}

function saveJobToStorage(job) {
    try {
        const arr = JSON.parse(localStorage.getItem('kh_jobs') || '[]');
        arr.push(job);
        localStorage.setItem('kh_jobs', JSON.stringify(arr));
        // add to runtime list so UI updates immediately
        JOBS.unshift(job);
    } catch (e) { console.error('Failed to save job', e); }
}

function deleteSavedJob(id) {
    try {
        let arr = JSON.parse(localStorage.getItem('kh_jobs') || '[]');
        arr = arr.filter(j => j.id !== id);
        localStorage.setItem('kh_jobs', JSON.stringify(arr));
        const idx = JOBS.findIndex(j => j.id === id);
        if (idx !== -1) JOBS.splice(idx, 1);
        // update UI if present
        const manageList = document.getElementById('manageJobsList');
        if (manageList) renderManageJobs();
        renderJobs(1);
    } catch (e) { console.error(e); }
}

// Edit saved job (simple prompt-based editor)
function editSavedJob(id) {
    try {
        let arr = JSON.parse(localStorage.getItem('kh_jobs') || '[]');
        const idx = arr.findIndex(j => j.id === id);
        if (idx === -1) { alert('Offre introuvable.'); return; }
        const job = arr[idx];
        const title = prompt('Titre', job.title) || job.title;
        const company = prompt('Entreprise', job.company) || job.company;
        const location = prompt('Localisation', job.location) || job.location;
        const salary = parseInt(prompt('Salaire', job.salary) || job.salary) || job.salary;
        const tags = (prompt('Tags (séparés par ,)', (job.tags||[]).join(',')) || (job.tags||[]).join(',')).split(',').map(s=>s.trim()).filter(Boolean);
        const desc = prompt('Description', job.desc) || job.desc;
        // update
        job.title = title; job.company = company; job.location = location; job.salary = salary; job.tags = tags; job.desc = desc;
        arr[idx] = job;
        localStorage.setItem('kh_jobs', JSON.stringify(arr));
        // update runtime JOBS
        const rIdx = JOBS.findIndex(j=>j.id===id);
        if (rIdx !== -1) JOBS[rIdx] = job;
        renderManageJobs();
        renderJobs(1);
        alert('Offre mise à jour.');
    } catch(e) { console.error(e); alert('Erreur lors de la modification'); }
}

// --- Applicants handling ---
function applyJob(id) {
    // if user is logged in and is job_seeker, save application
    const current = JSON.parse(localStorage.getItem('kh_current_user') || 'null');
    const job = JOBS.find(j=>j.id===id);
    if (!job) { alert('Offre introuvable'); return; }
    if (current && current.role === 'job_seeker') {
        const applicant = { id: Date.now(), jobId: id, applicantEmail: current.email, applicantName: current.email.split('@')[0], date: new Date().toISOString(), status: 'applied' };
        const apps = JSON.parse(localStorage.getItem('kh_applicants') || '[]');
        apps.push(applicant);
        localStorage.setItem('kh_applicants', JSON.stringify(apps));
        // update UI where possible
        try { renderApplicantsForJob(id, 'modalApplicants'); } catch(e) {}
        try { renderManageJobs(); } catch(e) {}
        try { renderCandidatesJobs(); } catch(e) {}
        alert('Votre candidature a été enregistrée.');
        return;
    }
    // if not logged in as job seeker, prompt for name/email and save
    const name = prompt('Votre nom (ou identifiant)');
    const email = prompt('Votre email');
    if (!email) { alert('Email requis pour postuler.'); return; }
    const applicant = { id: Date.now(), jobId: id, applicantEmail: email, applicantName: name || email.split('@')[0], date: new Date().toISOString(), status: 'applied' };
    const apps = JSON.parse(localStorage.getItem('kh_applicants') || '[]');
    apps.push(applicant);
    localStorage.setItem('kh_applicants', JSON.stringify(apps));
    // update UI
    try { renderApplicantsForJob(id, 'modalApplicants'); } catch(e) {}
    try { renderManageJobs(); } catch(e) {}
    try { renderCandidatesJobs(); } catch(e) {}
    alert('Candidature envoyée (demo).');
}

function getApplicants(jobId) {
    try { return JSON.parse(localStorage.getItem('kh_applicants') || '[]').filter(a=>a.jobId===jobId); } catch(e) { return []; }
}

function renderApplicantsForJob(jobId, containerId) {
    const el = document.getElementById(containerId);
    if (!el) return;
    const apps = getApplicants(jobId);
    if (!apps.length) { el.innerHTML = '<p class="placeholder">Aucun candidat pour cette offre.</p>'; return; }
    el.innerHTML = '<div class="applicants-list">' + apps.map(a=>`
        <div class="applicant-entry">
            <div>
                <div class="meta">${escapeHtml(a.applicantName)}</div>
                <div class="sub">${escapeHtml(a.applicantEmail)}</div>
            </div>
            <div style="text-align:right">
                <div class="sub">${new Date(a.date).toLocaleString()}</div>
            </div>
        </div>
    `).join('') + '</div>';
}

// Toggle applicants section (show/hide and render) for candidates page
function toggleApplicants(jobId, containerId) {
    const el = document.getElementById(containerId);
    if (!el) return;
    if (el.style.display && el.style.display !== 'none') {
        el.style.display = 'none';
        return;
    }
    el.style.display = 'block';
    try { renderApplicantsForJob(jobId, containerId); } catch(e) { el.innerHTML = '<p class="placeholder">Erreur lors du chargement des candidatures.</p>'; }
}

function renderManageJobs() {
    const listEl = document.getElementById('manageJobsList');
    if (!listEl) return;
    const arr = JSON.parse(localStorage.getItem('kh_jobs') || '[]');
    if (!arr.length) { listEl.innerHTML = '<p class="placeholder">Aucune offre créée par vous.</p>'; return; }
    listEl.innerHTML = arr.map(j => `
        <div class="job-card" style="min-height:80px; display:flex; align-items:center; gap:12px; padding:12px">
            <div style="flex:1">
                <strong>${escapeHtml(j.title || '')}</strong>
                <div class="job-company">${escapeHtml(j.company || '')} — ${escapeHtml(j.location || '')}</div>
                <div style="color:var(--text-light); font-size:0.85rem">Propriétaire: ${escapeHtml(j.owner || 'vous')}</div>
            </div>
            <div style="display:flex; gap:8px">
                <button class="card-btn view" onclick="openJobModal(${j.id})">Voir</button>
                <button class="card-btn" onclick="editSavedJob(${j.id})">Modifier</button>
                <button class="card-btn" onclick="deleteSavedJob(${j.id})">Supprimer</button>
            </div>
        </div>
    `).join('');
}

// Render recruiter jobs list for candidates page (reusable)
function renderCandidatesJobs() {
    try {
        if (typeof loadSavedJobs === 'function') loadSavedJobs();
        const arr = JSON.parse(localStorage.getItem('kh_jobs') || '[]');
        const listEl = document.getElementById('candidatesJobsList');
        if (!listEl) return;
        const current = JSON.parse(localStorage.getItem('kh_current_user') || 'null');
        const owned = current && current.role === 'recruiter' ? arr.filter(j => j.owner === current.email) : arr;
        if (!owned.length) {
            listEl.innerHTML = '<p class="placeholder">Vous n\'avez pas encore publié d\'offres. Créez-en une sur "Poster Offre".</p>';
            return;
        }
        listEl.innerHTML = owned.map(j => `
            <div style="margin-bottom:12px">
              <div class="job-card" style="display:flex;align-items:center;gap:12px;padding:12px">
                <div style="flex:1">
                    <strong>${escapeHtml(j.title || '')}</strong>
                    <div class="job-company">${escapeHtml(j.company || '')} — ${escapeHtml(j.location || '')}</div>
                    <div style="margin-top:6px"><span class="applicant-count">${getApplicants(j.id).length} candidatures</span></div>
                </div>
                <div style="display:flex;gap:8px">
                    <button class="card-btn view" onclick="openJobModal(${j.id})">Voir</button>
                    <button class="card-btn" onclick="window.location.href='manage_jobs.html'">Gérer</button>
                    <button class="card-btn" onclick="toggleApplicants(${j.id}, 'applicants-${j.id}')">Voir candidats</button>
                </div>
              </div>
              <div id="applicants-${j.id}" style="display:none; margin-top:6px; background: #fff; border-radius:8px; padding:8px; box-shadow: 0 6px 18px rgba(0,0,0,0.04)"></div>
            </div>
        `).join('');
    } catch (e) { console.warn('renderCandidatesJobs error', e); }
}

function escapeHtml(str) {
    return String(str || '').replace(/[&<>"']/g, function(m){ return ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[m]); });
}

const TYPE_BADGE = {
    'Full-time': 'badge-full',
    'Remote': 'badge-remote',
    'Part-time': 'badge-part',
    'CDI': 'badge-cdi'
};

function getFilters() {
    const qEl = document.getElementById('searchInput');
    const locEl = document.getElementById('filterLocation');
    const typeEl = document.getElementById('filterType');
    const sortEl = document.getElementById('sortBy');
    return {
        q: qEl ? String(qEl.value).trim().toLowerCase() : '',
        location: locEl ? String(locEl.value) : '',
        type: typeEl ? String(typeEl.value) : '',
        sort: sortEl ? String(sortEl.value) : ''
    };
}

function applyFilters(list) {
    const f = getFilters();
    let out = list.filter(j => {
        const matchQ = !f.q || (j.title + ' ' + j.company + ' ' + j.tags.join(' ')).toLowerCase().includes(f.q);
        const matchLoc = !f.location || j.location === f.location;
        const matchType = !f.type || j.type === f.type;
        return matchQ && matchLoc && matchType;
    });
    if (f.sort === 'salary_desc') out.sort((a,b)=>b.salary-a.salary);
    else if (f.sort === 'salary_asc') out.sort((a,b)=>a.salary-b.salary);
    else out.sort((a,b)=>b.id - a.id); // recent by id
    return out;
}

function toggleShowAll() {
    SHOW_ALL = !SHOW_ALL;
    const btn = document.getElementById('viewAllBtn');
    btn.classList.toggle('active', SHOW_ALL);
    btn.textContent = SHOW_ALL ? 'Voir moins' : 'Voir toutes';
    renderJobs(1);
}

function renderPagination(pages, current, total) {
    const pEl = document.getElementById('pagination');
    if (pages <= 1) { pEl.innerHTML = ''; return; }

    let html = '';
    // previous button
    html += `<button ${current===1? 'disabled':''} onclick="renderJobs(${current-1})"><i class="fas fa-chevron-left"></i></button>`;
    // pages window (max 7 buttons)
    const start = Math.max(1, current - 3);
    const end = Math.min(pages, start + 6);
    for (let i = start; i <= end; i++) {
        html += `<button ${i===current? 'disabled':''} onclick="renderJobs(${i})">${i}</button>`;
    }
    // next button
    html += `<button ${current===pages? 'disabled':''} onclick="renderJobs(${current+1})"><i class="fas fa-chevron-right"></i></button>`;
    html += `<div style="margin-left:12px; color:var(--text-light)">Résultats: ${total}</div>`;
    pEl.innerHTML = html;
}

function renderJobs(page = 1) {
    currentPage = page;
    const container = document.getElementById('jobsList');
    if (!container) return; // page doesn't have jobs list (avoid errors on analytics, candidates pages)
    const all = applyFilters(JOBS);
    const total = all.length;

    const pages = SHOW_ALL ? 1 : Math.max(1, Math.ceil(total / PAGE_SIZE));
    const start = SHOW_ALL ? 0 : (page - 1) * PAGE_SIZE;
    const slice = SHOW_ALL ? all : all.slice(start, start + PAGE_SIZE);

    container.innerHTML = slice.map(job => {
        const badgeClass = TYPE_BADGE[job.type] || 'badge-full';
        return `
        <article class="job-card" data-id="${job.id}">
            <div class="job-media">
                <img class="job-img" src="${job.img}" alt="${job.company}">
                <div class="salary-badge">${job.salary.toLocaleString('fr-FR')} €</div>
            </div>
            <div class="job-body">
                <div class="job-top">
                    <div>
                        <h3 class="job-title">${job.title}</h3>
                        <p class="job-company">${job.company} — ${job.location}</p>
                    </div>
                    <div class="job-badge ${badgeClass}">${job.type}</div>
                </div>
                <div class="job-meta">
                    <div class="job-tags">${job.tags.map(t=>`<span class="tag">${t}</span>`).join('')}</div>
                    <div class="job-actions">
                        <button class="card-btn view" onclick="openJobModal(${job.id})">Voir</button>
                        <button class="card-btn apply" onclick="applyJob(${job.id})">Postuler</button>
                    </div>
                </div>
            </div>
        </article>`;
    }).join('');

    
    if (SHOW_ALL) {
        document.getElementById('pagination').innerHTML = `<div style="color:var(--text-light); text-align:center; margin-top:12px">Affichage de toutes les offres (${total})</div>`;
    } else {
        renderPagination(pages, page, total);
    }
}


function openJobModal(id) {
    const job = JOBS.find(j=>j.id===id);
    if (!job) return;
    const modal = document.getElementById('jobModal');
    const body = document.getElementById('modalBody');
    body.innerHTML = `
        <div class="modal-header">
            <img class="logo-img" src="${job.img}" alt="${job.company}">
            <div class="meta">
                <h2>${job.title}</h2>
                <p>${job.company} · ${job.location} · <span style="font-weight:800">${job.type}</span></p>
            </div>
            <button class="modal-close" onclick="closeJobModal()">&times;</button>
        </div>
        <div class="modal-body">
            <div class="modal-salary">Salaire estimé: ${job.salary.toLocaleString('fr-FR')} €</div>
            <p>${job.desc}</p>
            <div class="modal-tags">${job.tags.map(t=>`<span class="tag">${t}</span>`).join('')}</div>
            <div class="modal-actions">
                <button class="apply-cta" onclick="applyJob(${job.id})">Postuler</button>
                <button class="close-cta" onclick="closeJobModal()">Fermer</button>
            </div>
            <hr style="margin:12px 0" />
            <div>
                <h3>Candidats</h3>
                <div id="modalApplicants"></div>
            </div>
        </div>
    `;
    modal.setAttribute('aria-hidden','false');
    // render applicants for this job into the modal
    try { renderApplicantsForJob(id, 'modalApplicants'); } catch(e) { console.warn('No applicants renderer', e); }
}

function closeJobModal() {
    const modal = document.getElementById('jobModal');
    modal.setAttribute('aria-hidden','true');
}
document.addEventListener('DOMContentLoaded', () => {
    ['searchInput','filterLocation','filterType','sortBy'].forEach(id=>{
        const el = document.getElementById(id);
        if (!el) return;
        el.addEventListener('input', ()=> renderJobs(1));
        el.addEventListener('change', ()=> renderJobs(1));
    });
    // load saved recruiter jobs first so they appear in listings
    loadSavedJobs();
    renderJobs(1);
    
    renderManageJobs();
    // If analytics canvas exists, try to render chart after a short delay
    try {
        const tryChart = () => {
            const canvas = document.getElementById('applicantsChart');
            if (!canvas) return;
            if (typeof renderApplicantsChart === 'function') {
                if (typeof Chart === 'undefined') {
                    // Chart library not available yet, retry shortly
                    setTimeout(tryChart, 150);
                    return;
                }
                try { renderApplicantsChart(); } catch(e) { console.warn('renderApplicantsChart error', e); }
            }
        };
        setTimeout(tryChart, 120);
    } catch(e) {}
});


document.addEventListener('keydown', (e)=> { if (e.key === 'Escape') closeJobModal(); });


function saveUserToStorage(user) {
    const users = JSON.parse(localStorage.getItem('kh_users') || '[]');
    users.push(user);
    localStorage.setItem('kh_users', JSON.stringify(users));
}

function findUser(email, password) {
    const users = JSON.parse(localStorage.getItem('kh_users') || '[]');
    return users.find(u => u.email === email && u.password === password);
}

// Handle registration form (register.html)
document.addEventListener('DOMContentLoaded', ()=>{
    const regForm = document.getElementById('registerForm');
    if (regForm) {
        regForm.addEventListener('submit', (ev)=>{
            ev.preventDefault();
            const form = ev.target;
            const email = form.email.value.trim();
            const password = form.password.value;
            const confirm = form.confirm_password.value;
            const role = document.querySelector('input[name="role"]:checked')?.value || 'job_seeker';

            if (!email || !password) { alert('Please fill email and password.'); return; }
            if (password !== confirm) { alert('Passwords do not match'); return; }

            // Save user (overwrite if exists)
            const users = JSON.parse(localStorage.getItem('kh_users') || '[]').filter(u=>u.email !== email);
            users.push({ email, password, role });
            localStorage.setItem('kh_users', JSON.stringify(users));

            // After registration, redirect to sign-in (login) page
            alert('Compte créé avec succès. Veuillez vous connecter.');
            window.location.href = 'login.html';
        });
    }

    // Handle login form (login.html)
    const loginForm = document.getElementById('loginForm');
    if (loginForm) {
        loginForm.addEventListener('submit', (ev)=>{
            ev.preventDefault();
            const form = ev.target;
            const email = document.getElementById('login_email')?.value || form.email?.value;
            const password = document.getElementById('login_password')?.value || form.password?.value;
            if (!email || !password) { alert('Please enter credentials'); return; }
            const user = findUser(email.trim(), password);
            if (!user) { alert('Invalid credentials (demo).'); return; }

            // simple session
            localStorage.setItem('kh_current_user', JSON.stringify({ email: user.email, role: user.role }));

            if (user.role === 'recruiter') window.location.href = 'recruiter_home.html';
            else window.location.href = 'index.html';
        });
    }
});

// Logout (deconnection)
function logout() {
    localStorage.removeItem('kh_current_user');
    // optionally keep users saved, just clear current session
    window.location.href = 'login.html';
}
