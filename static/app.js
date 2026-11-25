/**
 * pamostore fraud check // Moderation Workbench
 * Interactive Client Script (Vanilla JS - Zero Build Standard)
 */

// ---------------------------------------------------------------------------
// Synthetic Image Canvas Generators (Self-contained, no external asset dependencies)
// ---------------------------------------------------------------------------
function generateGarmentSampleImage() {
  const canvas = document.createElement('canvas');
  canvas.width = 224;
  canvas.height = 224;
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = '#F8FAFC';
  ctx.fillRect(0, 0, 224, 224);

  // Mannequin head and neck
  ctx.fillStyle = '#D4A373';
  ctx.beginPath();
  ctx.arc(112, 30, 15, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillRect(106, 45, 12, 15);

  // Bodice
  ctx.fillStyle = '#B4285A';
  ctx.beginPath();
  ctx.moveTo(92, 60);
  ctx.lineTo(132, 60);
  ctx.lineTo(124, 105);
  ctx.lineTo(100, 105);
  ctx.closePath();
  ctx.fill();

  // Flared kitenge skirt
  ctx.fillStyle = '#C8326E';
  ctx.beginPath();
  ctx.moveTo(100, 105);
  ctx.lineTo(124, 105);
  ctx.lineTo(165, 210);
  ctx.lineTo(59, 210);
  ctx.closePath();
  ctx.fill();

  // Golden kitenge pattern stripes
  ctx.strokeStyle = '#F59E0B';
  ctx.lineWidth = 3;
  for (let y = 115; y < 205; y += 15) {
    ctx.beginPath();
    ctx.moveTo(70 + (y - 105) * 0.3, y);
    ctx.lineTo(154 - (y - 105) * 0.3, y);
    ctx.stroke();
  }

  // Waist belt
  ctx.fillStyle = '#F59E0B';
  ctx.fillRect(98, 102, 28, 5);

  return canvas.toDataURL('image/jpeg', 0.90);
}

function generateCannabisSampleImage() {
  const canvas = document.createElement('canvas');
  canvas.width = 224;
  canvas.height = 224;
  const ctx = canvas.getContext('2d');

  // Neutral background
  ctx.fillStyle = '#F8FAFC';
  ctx.fillRect(0, 0, 224, 224);

  // Central stem
  ctx.strokeStyle = '#14532D';
  ctx.lineWidth = 4;
  ctx.beginPath();
  ctx.moveTo(112, 190);
  ctx.lineTo(112, 100);
  ctx.stroke();

  // Draw 7 cannabis palmate serrated leaf lobes radiating outward
  const angles = [-65, -45, -22, 0, 22, 45, 65];
  const lengths = [55, 75, 90, 100, 90, 75, 55];

  ctx.fillStyle = '#15803D';
  ctx.strokeStyle = '#166534';
  ctx.lineWidth = 1.5;

  for (let i = 0; i < angles.length; i++) {
    const rad = ((angles[i] - 90) * Math.PI) / 180;
    const len = lengths[i];
    const tipX = 112 + Math.cos(rad) * len;
    const tipY = 120 + Math.sin(rad) * len;

    const perpAngle = rad + Math.PI / 2;
    const halfWidth = 9;
    const midX = 112 + Math.cos(rad) * (len * 0.5);
    const midY = 120 + Math.sin(rad) * (len * 0.5);

    const cp1X = midX + Math.cos(perpAngle) * halfWidth;
    const cp1Y = midY + Math.sin(perpAngle) * halfWidth;
    const cp2X = midX - Math.cos(perpAngle) * halfWidth;
    const cp2Y = midY - Math.sin(perpAngle) * halfWidth;

    ctx.beginPath();
    ctx.moveTo(112, 120);
    ctx.quadraticCurveTo(cp1X, cp1Y, tipX, tipY);
    ctx.quadraticCurveTo(cp2X, cp2Y, 112, 120);
    ctx.fill();
    ctx.stroke();

    // Central leaf vein
    ctx.beginPath();
    ctx.strokeStyle = '#22C55E';
    ctx.lineWidth = 1;
    ctx.moveTo(112, 120);
    ctx.lineTo(tipX, tipY);
    ctx.stroke();
  }

  // Draw leafy bud clusters at base
  ctx.fillStyle = '#166534';
  ctx.beginPath();
  ctx.arc(112, 130, 16, 0, Math.PI * 2);
  ctx.fill();

  ctx.fillStyle = '#15803D';
  ctx.beginPath();
  ctx.arc(104, 138, 12, 0, Math.PI * 2);
  ctx.arc(120, 138, 12, 0, Math.PI * 2);
  ctx.fill();

  return canvas.toDataURL('image/jpeg', 0.90);
}

function generateMonochromeDummyImage() {
  const canvas = document.createElement('canvas');
  canvas.width = 32;
  canvas.height = 32;
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = '#050505';
  ctx.fillRect(0, 0, 32, 32);
  return canvas.toDataURL('image/jpeg', 0.8);
}

// ---------------------------------------------------------------------------
// Application State
// ---------------------------------------------------------------------------
const state = {
  currentImageBase64: generateGarmentSampleImage(),
  currentFileName: 'sinza_vitenge_sample.jpg',
  lastEvaluatedPayload: null,
  lastEvaluatedResult: null,
};

// ---------------------------------------------------------------------------
// DOM Elements
// ---------------------------------------------------------------------------
const el = {
  // Navigation & Metrics
  headerLatency: document.getElementById('headerLatency'),
  btnReloadLedger: document.getElementById('btnReloadLedger'),
  lastSyncTime: document.getElementById('lastSyncTime'),
  kpiInferences: document.getElementById('kpiInferences'),
  kpiFeedbacks: document.getElementById('kpiFeedbacks'),
  kpiAccuracy: document.getElementById('kpiAccuracy'),
  kpiF1: document.getElementById('kpiF1'),
  auditTableBody: document.getElementById('auditTableBody'),

  // Presets
  presetLegit: document.getElementById('presetLegit'),
  presetCharger: document.getElementById('presetCharger'),
  presetFlour: document.getElementById('presetFlour'),
  presetIllegal: document.getElementById('presetIllegal'),
  presetBorderline: document.getElementById('presetBorderline'),

  // Form Inputs
  simulationForm: document.getElementById('simulationForm'),
  listingId: document.getElementById('listingId'),
  sellerId: document.getElementById('sellerId'),
  productTitle: document.getElementById('productTitle'),
  productCategory: document.getElementById('productCategory'),
  productDescription: document.getElementById('productDescription'),
  priceTzs: document.getElementById('priceTzs'),
  formattedPrice: document.getElementById('formattedPrice'),
  hourOfDay: document.getElementById('hourOfDay'),
  hourDisplay: document.getElementById('hourDisplay'),
  accountAgeDays: document.getElementById('accountAgeDays'),
  completedOrders: document.getElementById('completedOrders'),
  disputeRate: document.getElementById('disputeRate'),
  repeatBuyer: document.getElementById('repeatBuyer'),
  btnRunEvaluation: document.getElementById('btnRunEvaluation'),
  evalSpinner: document.getElementById('evalSpinner'),
  evalBtnText: document.getElementById('evalBtnText'),

  // Image Upload Area
  dropZone: document.getElementById('dropZone'),
  fileInput: document.getElementById('fileInput'),
  dropContentEmpty: document.getElementById('dropContentEmpty'),
  dropContentPreview: document.getElementById('dropContentPreview'),
  imagePreview: document.getElementById('imagePreview'),
  imageFileName: document.getElementById('imageFileName'),
  imageFileSize: document.getElementById('imageFileSize'),
  btnRemoveImage: document.getElementById('btnRemoveImage'),

  // Verdict & Telemetry
  verdictCard: document.getElementById('verdictCard'),
  verdictBadge: document.getElementById('verdictBadge'),
  verdictSummary: document.getElementById('verdictSummary'),
  verdictLatency: document.getElementById('verdictLatency'),
  fraudProbText: document.getElementById('fraudProbText'),
  riskReasonsContainer: document.getElementById('riskReasonsContainer'),
  riskReasonsList: document.getElementById('riskReasonsList'),

  gaugeCategoryInfo: document.getElementById('gaugeCategoryInfo'),
  gaugeCategoryDesc: document.getElementById('gaugeCategoryDesc'),
  gaugeSlangScore: document.getElementById('gaugeSlangScore'),
  gaugeSlangBar: document.getElementById('gaugeSlangBar'),
  matchedSlangPills: document.getElementById('matchedSlangPills'),
  gaugeMismatchScore: document.getElementById('gaugeMismatchScore'),
  gaugeMismatchBar: document.getElementById('gaugeMismatchBar'),
  gaugeComparisonBadge: document.getElementById('gaugeComparisonBadge'),
  gaugeClaimedConcept: document.getElementById('gaugeClaimedConcept'),
  gaugeTopVisual: document.getElementById('gaugeTopVisual'),
  gaugeTopVisualProb: document.getElementById('gaugeTopVisualProb'),
  gaugeContrabandScore: document.getElementById('gaugeContrabandScore'),
  gaugeContrabandBar: document.getElementById('gaugeContrabandBar'),
  gaugeRiverScore: document.getElementById('gaugeRiverScore'),
  gaugeRiverBar: document.getElementById('gaugeRiverBar'),

  // Online Learning Feedback
  btnFeedbackLegit: document.getElementById('btnFeedbackLegit'),
  btnFeedbackFraud: document.getElementById('btnFeedbackFraud'),
  feedbackResultBanner: document.getElementById('feedbackResultBanner'),
  shiftDelta: document.getElementById('shiftDelta'),
  shiftDetails: document.getElementById('shiftDetails'),
  btnReRunScenario: document.getElementById('btnReRunScenario'),
};

// ---------------------------------------------------------------------------
// Formatters & Utilities
// ---------------------------------------------------------------------------
function formatTzs(num) {
  return Number(num).toLocaleString('en-US') + ' TZS';
}

function formatHourLabel(hour) {
  const h = parseInt(hour, 10);
  const period = h >= 12 ? 'PM' : 'AM';
  const displayHour = h % 12 === 0 ? 12 : h % 12;
  const timeTag = (h >= 1 && h <= 5) ? ' (Night Alert)' : '';
  return `${String(h).padStart(2, '0')}:00 (${displayHour}:00 ${period})${timeTag}`;
}

// ---------------------------------------------------------------------------
// Image Preview Handlers
// ---------------------------------------------------------------------------
function showImagePreview(base64Data, filename, sizeLabel) {
  state.currentImageBase64 = base64Data;
  state.currentFileName = filename;
  el.imagePreview.src = base64Data;
  el.imageFileName.textContent = filename;
  el.imageFileSize.textContent = sizeLabel || 'Attached Media';
  el.dropContentEmpty.classList.add('hidden');
  el.dropContentPreview.classList.remove('hidden');
}

function clearImagePreview() {
  state.currentImageBase64 = null;
  state.currentFileName = null;
  el.fileInput.value = '';
  el.dropContentEmpty.classList.remove('hidden');
  el.dropContentPreview.classList.add('hidden');
}

// ---------------------------------------------------------------------------
// Presets Configuration (Tanzanian Retail Scenarios)
// ---------------------------------------------------------------------------
const presets = {
  legit: {
    listing_id: 'LIST-SINZA-1001',
    seller_id: 'USR-SINZA-FASHION',
    title: 'Gauni Nzuri ya Vitenge vya Kisasa Kariakoo',
    category: 'fashion',
    description: 'Nguo bora ya pamba, rangi ya kuvutia, inafaa kwa harusi na sherehe Sinza. Delivery na rider ipo.',
    price_tzs: 45000,
    hour_of_day: 14,
    account_age_days: 180,
    seller_completed_orders: 85,
    seller_dispute_rate: 0.01,
    is_repeat_buyer_same_day: 0,
    image_base64: generateGarmentSampleImage(),
    image_name: 'kariakoo_kitenge.jpg',
    image_label: 'Garment Textile Image (High Correlation)',
  },
  charger: {
    listing_id: 'LIST-MWENGE-CHG-01',
    seller_id: 'USR-MWENGE-ELECTRONICS',
    title: 'Chaja Original ya Simu Fast Charging Type-C',
    category: 'electronics',
    description: 'Fast charger Type-C mpya kabisa dukani Mwenge, inachaji simu zote za kisasa haraka na salama.',
    price_tzs: 35000,
    hour_of_day: 14,
    account_age_days: 90,
    seller_completed_orders: 45,
    seller_dispute_rate: 0.01,
    is_repeat_buyer_same_day: 0,
    image_base64: generateGarmentSampleImage(),
    image_name: 'charger_original.jpg',
    image_label: 'Electronics Product Media',
  },
  flour: {
    listing_id: 'LIST-KARI-SEMBE-02',
    seller_id: 'USR-GROCERY-STORE',
    title: 'Unga wa Sembe Bora Kilo 5',
    category: 'groceries',
    description: 'Unga safi wa sembe namba moja kwa ugali laini na mtamu, kilo 5 kwa bei ya jumla Kariakoo.',
    price_tzs: 15000,
    hour_of_day: 11,
    account_age_days: 120,
    seller_completed_orders: 60,
    seller_dispute_rate: 0.00,
    is_repeat_buyer_same_day: 0,
    image_base64: generateGarmentSampleImage(),
    image_name: 'unga_wa_sembe_5kg.jpg',
    image_label: 'Grocery Grain Media',
  },
  illegal: {
    listing_id: 'LIST-DISGUISED-992',
    seller_id: 'USR-SHADY-007',
    title: 'Nauza gauni la harusi bei poa sana',
    category: 'fashion',
    description: 'Mzigo mzito wa cha arusha na bangi freshi, msokoto kete za kutosha, wasiliana haraka kwa namba ya siri.',
    price_tzs: 450000,
    hour_of_day: 3,
    account_age_days: 1,
    seller_completed_orders: 0,
    seller_dispute_rate: 0.35,
    is_repeat_buyer_same_day: 1,
    image_base64: generateCannabisSampleImage(),
    image_name: 'disguised_bangi_mzigo.jpg',
    image_label: 'Serrated Cannabis Foliage (Prohibited Contraband)',
  },
  borderline: {
    listing_id: 'LIST-BORDERLINE-404',
    seller_id: 'USR-MWENGE-TECH',
    title: 'Simu Mpya ya Android Dar es Salaam',
    category: 'electronics',
    description: 'Simu mpya kabisa haijafunguliwa, risiti ipo kutoka duka la Mwenge.',
    price_tzs: 120000,
    hour_of_day: 13,
    account_age_days: 45,
    seller_completed_orders: 8,
    seller_dispute_rate: 0.06,
    is_repeat_buyer_same_day: 0,
    image_base64: generateGarmentSampleImage(),
    image_name: 'phone_accessory_sample.jpg',
    image_label: 'Electronic Device Photo',
  },
};

function loadPreset(presetKey) {
  const p = presets[presetKey];
  if (!p) return;

  el.listingId.value = p.listing_id;
  el.sellerId.value = p.seller_id;
  el.productTitle.value = p.title;
  el.productCategory.value = p.category || '';
  el.productDescription.value = p.description;
  el.priceTzs.value = p.price_tzs;
  el.formattedPrice.textContent = '~ ' + formatTzs(p.price_tzs);
  el.hourOfDay.value = p.hour_of_day;
  el.hourDisplay.textContent = formatHourLabel(p.hour_of_day);
  el.accountAgeDays.value = p.account_age_days;
  el.completedOrders.value = p.seller_completed_orders;
  el.disputeRate.value = p.seller_dispute_rate;
  el.repeatBuyer.value = String(p.is_repeat_buyer_same_day);

  showImagePreview(p.image_base64, p.image_name, p.image_label);

  runEvaluation();
}

// ---------------------------------------------------------------------------
// Run Dual-Layer Evaluation
// ---------------------------------------------------------------------------
async function runEvaluation() {
  const payload = {
    listing_id: el.listingId.value.trim() || 'LIST-DEMO-01',
    seller_id: el.sellerId.value.trim() || 'USR-DEMO-01',
    title: el.productTitle.value.trim(),
    category: el.productCategory.value || undefined,
    description: el.productDescription.value.trim(),
    price_tzs: parseFloat(el.priceTzs.value) || 10000,
    account_age_days: parseFloat(el.accountAgeDays.value) || 0,
    seller_completed_orders: parseInt(el.completedOrders.value, 10) || 0,
    seller_dispute_rate: parseFloat(el.disputeRate.value) || 0.0,
    hour_of_day: parseInt(el.hourOfDay.value, 10) || 12,
    is_repeat_buyer_same_day: parseInt(el.repeatBuyer.value, 10) || 0,
    image_base64: state.currentImageBase64,
  };

  state.lastEvaluatedPayload = payload;

  el.btnRunEvaluation.disabled = true;
  el.evalSpinner.classList.remove('hidden');
  el.evalBtnText.textContent = 'Evaluating Dual-Layer Radar...';

  const startTime = performance.now();

  try {
    const res = await fetch('/api/v1/moderate-listing', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });

    const elapsed = Math.round(performance.now() - startTime);
    el.headerLatency.textContent = `${elapsed} ms`;
    el.verdictLatency.textContent = `Response: ${elapsed} ms`;

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Evaluation request failed');
    }

    const data = await res.json();
    state.lastEvaluatedResult = data;
    renderVerdict(data);
    loadAuditLedger();
  } catch (error) {
    console.error('Error running evaluation:', error);
    alert('Evaluation error: ' + error.message);
  } finally {
    el.btnRunEvaluation.disabled = false;
    el.evalSpinner.classList.add('hidden');
    el.evalBtnText.textContent = 'Run Dual-Layer Risk Evaluation';
  }
}

// ---------------------------------------------------------------------------
// Render Verdict & Dual-Layer Gauges
// ---------------------------------------------------------------------------
function renderVerdict(data) {
  const { decision, fraud_probability, risk_reasons, layer1_details, category, price_to_category_median_ratio } = data;
  const pct = (fraud_probability * 100).toFixed(1);

  el.fraudProbText.textContent = `${pct}%`;

  // Verdict Badge Styling (High contrast, unslop Linear/Stripe styling)
  if (decision === 'ALLOW') {
    el.verdictBadge.textContent = 'ALLOW';
    el.verdictBadge.className = 'inline-block px-3 py-1 rounded text-sm font-bold tracking-wide font-mono bg-emerald-50 text-emerald-700 border border-emerald-200';
    el.verdictSummary.textContent = 'Standard transaction pattern. Funds cleared for escrow lock.';
    el.fraudProbText.className = 'text-2xl font-bold font-mono text-emerald-700';
  } else if (decision === 'HOLD_FOR_REVIEW') {
    el.verdictBadge.textContent = 'HOLD FOR REVIEW';
    el.verdictBadge.className = 'inline-block px-3 py-1 rounded text-sm font-bold tracking-wide font-mono bg-amber-50 text-amber-700 border border-amber-200';
    el.verdictSummary.textContent = 'Elevated transaction anomaly detected. Rider dispatch paused pending OTP or manual audit.';
    el.fraudProbText.className = 'text-2xl font-bold font-mono text-amber-700';
  } else {
    el.verdictBadge.textContent = 'BLOCK';
    el.verdictBadge.className = 'inline-block px-3 py-1 rounded text-sm font-bold tracking-wide font-mono bg-rose-50 text-rose-700 border border-rose-200';
    el.verdictSummary.textContent = 'Prohibited goods or critical escrow risk. Listing suppressed; merchant locked.';
    el.fraudProbText.className = 'text-2xl font-bold font-mono text-rose-700';
  }

  // Risk Reasons
  if (risk_reasons && risk_reasons.length > 0) {
    el.riskReasonsContainer.classList.remove('hidden');
    el.riskReasonsList.innerHTML = risk_reasons.map(r => `<li>${r}</li>`).join('');
  } else {
    el.riskReasonsContainer.classList.add('hidden');
    el.riskReasonsList.innerHTML = '';
  }

  // Category Benchmark Telemetry
  if (el.gaugeCategoryInfo && category) {
    const ratioStr = price_to_category_median_ratio !== undefined ? `${price_to_category_median_ratio.toFixed(2)}x` : '1.00x';
    el.gaugeCategoryInfo.textContent = `${category.toUpperCase()} (${ratioStr} median)`;
    if (el.gaugeCategoryDesc) {
      el.gaugeCategoryDesc.textContent = `Auto-benchmarked against Tanzanian ${category} market median | Normalized ratio: ${ratioStr}`;
    }
  }

  // Layer 1 Slang Telemetry
  const slangScore = layer1_details.swahili_slang_risk_score || 0.0;
  el.gaugeSlangScore.textContent = slangScore.toFixed(2);
  el.gaugeSlangBar.style.width = `${Math.min(100, slangScore * 100)}%`;
  el.gaugeSlangBar.className = slangScore > 0.6 ? 'h-full bg-rose-500 transition-all duration-300' : (slangScore > 0.3 ? 'h-full bg-amber-500 transition-all duration-300' : 'h-full bg-emerald-500 transition-all duration-300');

  // Matched slang pills
  const terms = layer1_details.matched_slang_terms || [];
  if (terms.length > 0) {
    el.matchedSlangPills.innerHTML = terms.map(t =>
      `<span class="px-2 py-0.5 rounded text-[11px] font-mono bg-rose-100 text-rose-800 border border-rose-200">#${t}</span>`
    ).join('');
  } else {
    el.matchedSlangPills.innerHTML = '<span class="text-[10px] text-slate-400 italic">No prohibited slang detected</span>';
  }

  // Layer 1 Mismatch Telemetry
  const mismatchScore = layer1_details.image_text_mismatch_score || 0.0;
  el.gaugeMismatchScore.textContent = mismatchScore.toFixed(2);
  el.gaugeMismatchBar.style.width = `${Math.min(100, mismatchScore * 100)}%`;
  el.gaugeMismatchBar.className = mismatchScore > 0.6 ? 'h-full bg-rose-500 transition-all duration-300' : (mismatchScore > 0.35 ? 'h-full bg-amber-500 transition-all duration-300' : 'h-full bg-emerald-500 transition-all duration-300');

  // Layer 1 Zero-Shot Visual Classification & Telemetry
  const topVisual = layer1_details.top_visual_label || data.top_visual_label || 'Safe Product Media';
  const rawProb = layer1_details.top_visual_probability !== undefined ? layer1_details.top_visual_probability : data.top_visual_probability;
  const topProb = ((rawProb || 0.0) * 100).toFixed(1);
  const claimedConcept = layer1_details.seller_claimed_concept || data.seller_claimed_concept || el.productTitle.value || 'Retail Merchandise';
  const comparisonStatus = layer1_details.visual_comparison_status || data.visual_comparison_status || (mismatchScore > 0.65 ? 'MISMATCH' : 'MATCH');
  const contrabandProb = ((layer1_details.visual_contraband_probability || 0.0) * 100).toFixed(1);
  const isVisualContraband = !!(layer1_details.visual_contraband_detected || data.visual_contraband_detected);

  if (el.gaugeClaimedConcept) {
    el.gaugeClaimedConcept.textContent = claimedConcept;
    el.gaugeClaimedConcept.title = claimedConcept;
  }
  if (el.gaugeTopVisual) {
    el.gaugeTopVisual.textContent = topVisual;
    el.gaugeTopVisual.title = topVisual;
  }
  if (el.gaugeTopVisualProb) {
    el.gaugeTopVisualProb.textContent = `(${topProb}%)`;
  }
  if (el.gaugeContrabandScore) {
    el.gaugeContrabandScore.textContent = `${contrabandProb}%`;
  }
  if (el.gaugeContrabandBar) {
    el.gaugeContrabandBar.style.width = `${Math.min(100, (layer1_details.visual_contraband_probability || 0) * 100)}%`;
    el.gaugeContrabandBar.className = isVisualContraband
      ? 'h-full bg-rose-500 transition-all duration-300'
      : (parseFloat(contrabandProb) > 20.0 ? 'h-full bg-amber-500 transition-all duration-300' : 'h-full bg-emerald-500 transition-all duration-300');
  }
  if (el.gaugeComparisonBadge) {
    if (comparisonStatus === 'MATCH') {
      el.gaugeComparisonBadge.textContent = 'MATCH';
      el.gaugeComparisonBadge.className = 'px-2 py-0.5 rounded text-[10px] font-mono font-bold bg-emerald-50 text-emerald-700 border border-emerald-200';
    } else if (comparisonStatus === 'MISMATCH') {
      el.gaugeComparisonBadge.textContent = 'MISMATCH';
      el.gaugeComparisonBadge.className = 'px-2 py-0.5 rounded text-[10px] font-mono font-bold bg-amber-50 text-amber-700 border border-amber-200';
    } else if (comparisonStatus === 'CONTRABAND') {
      el.gaugeComparisonBadge.textContent = 'CONTRABAND';
      el.gaugeComparisonBadge.className = 'px-2 py-0.5 rounded text-[10px] font-mono font-bold bg-rose-50 text-rose-700 border border-rose-200';
    } else {
      el.gaugeComparisonBadge.textContent = comparisonStatus;
      el.gaugeComparisonBadge.className = 'px-2 py-0.5 rounded text-[10px] font-mono font-bold bg-slate-100 text-slate-600 border border-slate-200';
    }
  }

  // Layer 2 River ML Telemetry
  const riverProb = fraud_probability || 0.0;
  el.gaugeRiverScore.textContent = riverProb.toFixed(4);
  el.gaugeRiverBar.style.width = `${Math.min(100, riverProb * 100)}%`;
  el.gaugeRiverBar.className = riverProb > 0.75 ? 'h-full bg-rose-500 transition-all duration-300' : (riverProb > 0.40 ? 'h-full bg-amber-500 transition-all duration-300' : 'h-full bg-emerald-500 transition-all duration-300');

  el.feedbackResultBanner.classList.add('hidden');
}

// ---------------------------------------------------------------------------
// Online Learning Feedback Submission (.learn_one)
// ---------------------------------------------------------------------------
async function submitFeedback(actualLabel) {
  if (!state.lastEvaluatedPayload || !state.lastEvaluatedResult) {
    alert('Please run an evaluation first before providing feedback.');
    return;
  }

  const features = state.lastEvaluatedResult.layer2_features;
  const txId = state.lastEvaluatedPayload.listing_id || 'TX-FEEDBACK-01';

  try {
    const res = await fetch('/api/v1/feedback', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        transaction_id: txId,
        actual_label: actualLabel,
        features: features,
        comment: actualLabel === 1
          ? 'Rider flagged counterfeit goods or contraband'
          : 'Buyer confirmed legitimate delivery via OTP',
      }),
    });

    if (!res.ok) {
      throw new Error('Feedback submission failed');
    }

    const data = await res.json();
    const prevPct = (data.previous_probability * 100).toFixed(2);
    const newPct = (data.updated_probability * 100).toFixed(2);
    const deltaPct = ((data.updated_probability - data.previous_probability) * 100).toFixed(2);
    const sign = deltaPct >= 0 ? '+' : '';

    el.shiftDelta.textContent = `${sign}${deltaPct}% shift`;
    el.shiftDelta.className = actualLabel === 1
      ? 'font-mono text-xs font-semibold text-rose-600'
      : 'font-mono text-xs font-semibold text-emerald-600';
    el.shiftDetails.textContent = `${prevPct}% → ${newPct}% probability (${actualLabel === 1 ? 'Fraud reinforcement' : 'Legit reinforcement'})`;
    el.feedbackResultBanner.classList.remove('hidden');

    loadAuditLedger();
  } catch (error) {
    console.error('Error submitting feedback:', error);
    alert('Failed to submit online learning feedback: ' + error.message);
  }
}

// ---------------------------------------------------------------------------
// Load Live Audit Ledger from SQLite (audit_logs.db)
// ---------------------------------------------------------------------------
async function loadAuditLedger() {
  try {
    const res = await fetch('/api/v1/audit-stats');
    if (!res.ok) throw new Error('Failed to fetch audit stats');

    const data = await res.json();
    const { metrics, recent_logs } = data;

    el.kpiInferences.textContent = metrics.total_inferences.toLocaleString();
    el.kpiFeedbacks.textContent = metrics.total_feedbacks.toLocaleString();
    el.kpiAccuracy.textContent = `${(metrics.running_accuracy * 100).toFixed(2)}%`;
    el.kpiF1.textContent = `${(metrics.running_f1 * 100).toFixed(2)}%`;
    el.lastSyncTime.textContent = `Synced: ${new Date().toLocaleTimeString()}`;

    if (!recent_logs || recent_logs.length === 0) {
      el.auditTableBody.innerHTML = `
        <tr><td colspan="8" class="text-center py-6 text-slate-400">No audit logs recorded yet.</td></tr>
      `;
      return;
    }

    el.auditTableBody.innerHTML = recent_logs.map(log => {
      let badgeClass = 'bg-slate-100 text-slate-700 border-slate-200';
      if (log.decision === 'ALLOW') badgeClass = 'bg-emerald-50 text-emerald-700 border-emerald-200';
      if (log.decision === 'HOLD_FOR_REVIEW') badgeClass = 'bg-amber-50 text-amber-700 border-amber-200';
      if (log.decision === 'BLOCK') badgeClass = 'bg-rose-50 text-rose-700 border-rose-200';

      const probFormatted = (log.fraud_probability * 100).toFixed(1) + '%';
      const formattedDate = log.created_at ? log.created_at.replace('T', ' ').slice(0, 19) : '--';

      return `
        <tr class="hover:bg-slate-50/80 transition">
          <td class="py-2.5 px-4 font-mono text-[11px] font-medium text-slate-600">${log.event_type}</td>
          <td class="py-2.5 px-4 font-mono text-[11px] text-slate-900 font-semibold">${log.reference_id}</td>
          <td class="py-2.5 px-4 font-mono text-[11px] text-slate-600">${log.seller_id}</td>
          <td class="py-2.5 px-4 text-slate-800 max-w-[240px] truncate" title="${log.detail}">${log.detail}</td>
          <td class="py-2.5 px-4 font-mono text-right text-slate-900">${formatTzs(log.price_tzs)}</td>
          <td class="py-2.5 px-4 font-mono text-right font-semibold text-slate-800">${probFormatted}</td>
          <td class="py-2.5 px-4 text-center">
            <span class="inline-block px-2 py-0.5 rounded text-[10px] font-mono font-bold border ${badgeClass}">
              ${log.decision}
            </span>
          </td>
          <td class="py-2.5 px-4 font-mono text-[11px] text-right text-slate-400">${formattedDate}</td>
        </tr>
      `;
    }).join('');

  } catch (error) {
    console.error('Error loading audit ledger:', error);
  }
}

// ---------------------------------------------------------------------------
// Event Listeners Initialization
// ---------------------------------------------------------------------------
function initEvents() {
  el.priceTzs.addEventListener('input', () => {
    // Dynamically set step='any' if decimal is entered, ensuring no browser step mismatch popups
    if (el.priceTzs.value.includes('.')) {
      el.priceTzs.setAttribute('step', 'any');
    } else {
      el.priceTzs.setAttribute('step', '1');
    }
    const val = parseFloat(el.priceTzs.value) || 0;
    el.formattedPrice.textContent = '~ ' + formatTzs(val);
  });

  el.hourOfDay.addEventListener('input', () => {
    el.hourDisplay.textContent = formatHourLabel(el.hourOfDay.value);
  });

  el.simulationForm.addEventListener('submit', (e) => {
    e.preventDefault();
    runEvaluation();
  });

  // Presets
  el.presetLegit.addEventListener('click', () => loadPreset('legit'));
  if (el.presetCharger) el.presetCharger.addEventListener('click', () => loadPreset('charger'));
  if (el.presetFlour) el.presetFlour.addEventListener('click', () => loadPreset('flour'));
  el.presetIllegal.addEventListener('click', () => loadPreset('illegal'));
  el.presetBorderline.addEventListener('click', () => loadPreset('borderline'));

  // Drag and Drop Image Handlers
  el.dropZone.addEventListener('click', () => el.fileInput.click());
  el.fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (event) => {
      const sizeKb = (file.size / 1024).toFixed(1) + ' KB';
      showImagePreview(event.target.result, file.name, sizeKb);
    };
    reader.readAsDataURL(file);
  });

  el.dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    el.dropZone.classList.add('border-slate-800', 'bg-slate-100/50');
  });

  el.dropZone.addEventListener('dragleave', () => {
    el.dropZone.classList.remove('border-slate-800', 'bg-slate-100/50');
  });

  el.dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    el.dropZone.classList.remove('border-slate-800', 'bg-slate-100/50');
    const file = e.dataTransfer.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (event) => {
      const sizeKb = (file.size / 1024).toFixed(1) + ' KB';
      showImagePreview(event.target.result, file.name, sizeKb);
    };
    reader.readAsDataURL(file);
  });

  el.btnRemoveImage.addEventListener('click', (e) => {
    e.stopPropagation();
    clearImagePreview();
  });

  // Feedback Buttons
  el.btnFeedbackLegit.addEventListener('click', () => submitFeedback(0));
  el.btnFeedbackFraud.addEventListener('click', () => submitFeedback(1));

  // Re-Run Scenario Button
  el.btnReRunScenario.addEventListener('click', () => {
    runEvaluation();
  });

  // Sync Ledger button
  el.btnReloadLedger.addEventListener('click', () => {
    loadAuditLedger();
  });
}

// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------
document.addEventListener('DOMContentLoaded', () => {
  initEvents();
  loadPreset('legit');
  loadAuditLedger();
});
