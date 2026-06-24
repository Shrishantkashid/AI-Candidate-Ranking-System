import json
import gzip
import pandas as pd
import numpy as np
from datetime import datetime, timezone
import os
import argparse

# Configuration arrays
TITLE_KEYWORDS = [
    "recommendation", "ranking", "retrieval", "search", "nlp", 
    "machine learning engineer", "ai engineer", "data scientist", 
    "ml engineer", "applied scientist", "research scientist"
]

ATS_JD_KEYWORDS = [
    "python", "machine learning", "nlp", "fastapi", "vector search", 
    "embeddings", "retrieval", "recommendation system"
]

ATS_STRONG_TERMS = [
    "developed", "implemented", "deployed", "optimized", "engineered", 
    "built", "production", "scaled", "designed"
]

ATS_ROLES = [
    "ai engineer", "machine learning engineer", "ml engineer", "nlp engineer"
]

REC_EVIDENCE_TERMS = [
    "built", "developed", "deployed", "production", "designed", 
    "scaled", "implemented", "shipped", "optimized", "launched"
]

REC_WEAK_TERMS = [
    "learned", "tutorial", "basic", "experiment", "demo", "practice"
]

REC_TOP_COMPANIES = [
    "google", "amazon", "microsoft", "meta", "flipkart", 
    "swiggy", "uber", "adobe", "atlassian", "netflix"
]

REC_EDU_TERMS = ["computer science", "artificial intelligence", "data science", "machine learning"]

PROD_ML_KEYWORDS = [
    "embedding", "vector search", "faiss", "pinecone", "weaviate", 
    "qdrant", "milvus", "hybrid retrieval", "learning to rank", 
    "xgboost ranking", "ndcg", "a/b test", "offline-online", 
    "ranking model", "retrieval system", "search relevance", 
    "recommendation system"
]

HIGH_VALUE_SKILLS = [
    "pinecone", "faiss", "weaviate", "milvus", "qdrant", 
    "sentence transformers", "information retrieval", 
    "learning to rank", "bm25", "elasticsearch", "opensearch", 
    "rag", "embeddings", "vector search"
]

CONSULTING_COMPANIES = {
    "tcs", "infosys", "wipro", "accenture", "cognizant", 
    "capgemini", "hcl", "mindtree", "tech mahindra", "mphasis", 
    "lti", "ust"
}

PRODUCT_COMPANIES = {
    "flipkart", "swiggy", "razorpay", "cred", "zomato", 
    "uber", "amazon", "google", "microsoft", "meta", 
    "apple", "netflix", "adobe", "atlassian", "intuit",
    "ola", "paytm", "myntra", "nykaa", "upgrad", 
    "vedantu", "unacademy"
}

def safe_get(d, keys, default=None):
    for key in keys:
        if isinstance(d, dict) and key in d:
            d = d[key]
        else:
            return default
    return d

def extract_title_relevance(profile, career_history):
    matched_keywords = set()
    current_title = str(profile.get("current_title", "")).lower()
    
    titles = [current_title]
    for job in career_history:
        titles.append(str(job.get("title", "")).lower())
    
    for kw in TITLE_KEYWORDS:
        for t in titles:
            if kw in t:
                matched_keywords.add(kw)
                break
                
    score = len(matched_keywords) / len(TITLE_KEYWORDS)
    
    current_title_match = any(kw in current_title for kw in TITLE_KEYWORDS)
    if current_title_match:
        score *= 1.2
        
    return min(1.0, score), ",".join(matched_keywords)

def extract_prod_ml_experience(career_history):
    decayed_prod_ml_months = 0
    raw_prod_ml_months = 0
    has_prod_boost = False
    
    import re
    def get_recency_multiplier(end_date_str, current_year=2025):
        if not end_date_str or str(end_date_str).lower() in ["present", "now", "current"]:
            return 1.0
        try:
            match = re.search(r'\b(20\d{2})\b', str(end_date_str))
            if match:
                year = int(match.group(1))
            else:
                return 0.5
            diff = current_year - year
            if diff <= 0: return 1.0
            elif diff == 1: return 0.9
            elif diff == 2: return 0.75
            elif diff == 3: return 0.6
            elif diff == 4: return 0.4
            else: return 0.2
        except:
            return 0.5
            
    for job in career_history:
        desc = str(job.get("description", "")).lower()
        duration = float(job.get("duration_months", 0))
        end_date = str(job.get("end_date", ""))
        
        has_keyword = False
        for kw in PROD_ML_KEYWORDS:
            if kw in desc:
                has_keyword = True
                break
                
        if has_keyword:
            raw_prod_ml_months += duration
            decay_mult = get_recency_multiplier(end_date)
            decayed_prod_ml_months += (duration * decay_mult)
            
            if "production" in desc or "deployed" in desc:
                has_prod_boost = True
                
    score = min(1.0, decayed_prod_ml_months / 36.0)
    if has_prod_boost:
        score += 0.2
        
    return min(1.0, score), raw_prod_ml_months / 12.0

def extract_skill_relevance(skills):
    total_contribution = 0.0
    has_langchain = False
    has_high_value = False
    matched_high_value_skills = []
    
    prof_map = {
        "beginner": 0.2,
        "intermediate": 0.6,
        "advanced": 0.9,
        "expert": 1.0
    }
    
    for skill in skills:
        name = str(skill.get("name", "")).lower()
        
        if name == "langchain":
            has_langchain = True
            
        if name in HIGH_VALUE_SKILLS:
            has_high_value = True
            matched_high_value_skills.append(skill.get("name", ""))
            
            prof_str = str(skill.get("proficiency", "")).lower()
            prof_score = prof_map.get(prof_str, 0.5)
            
            endorsements = float(skill.get("endorsements", 0))
            end_norm = min(1.0, endorsements / 100.0)
            
            duration = float(skill.get("duration_months", 0))
            dur_norm = min(1.0, duration / 24.0)
            
            contrib = prof_score * (0.5 + 0.5 * end_norm) * (0.5 + 0.5 * dur_norm)
            total_contribution += contrib
            
    score = min(1.0, total_contribution / 5.0)
    
    if has_langchain and not has_high_value:
        score = max(0.0, score - 0.3)
        
    return score, ",".join(matched_high_value_skills)

def is_consulting_only(career_history):
    if not career_history:
        return 0.0
        
    all_consulting = True
    for job in career_history:
        company = str(job.get("company", "")).lower()
        if any(prod_comp in company for prod_comp in PRODUCT_COMPANIES):
            return 0.0
            
        if not any(cons_comp in company for cons_comp in CONSULTING_COMPANIES):
            all_consulting = False
            
    return 0.7 if all_consulting else 0.0

def is_research_only(career_history, prod_ml_score):
    for job in career_history:
        title = str(job.get("title", "")).lower()
        if "research scientist" in title or "researcher" in title:
            if prod_ml_score == 0:
                return 0.5
    return 0.0

def compute_title_hopping_penalty(career_history):
    if not career_history:
        return 0.0
        
    total_months = 0
    count = 0
    
    for job in career_history:
        duration = job.get("duration_months")
        if duration is not None and float(duration) > 0:
            total_months += float(duration)
            count += 1
            
    if count == 0:
        return 0.0
        
    avg_tenure = total_months / count
    return 0.2 if avg_tenure < 18 else 0.0

def compute_langchain_only_penalty(skills, skill_relevance_score):
    has_langchain = any(str(s.get("name", "")).lower() == "langchain" for s in skills)
    if has_langchain and skill_relevance_score < 0.3:
        return 0.4
    return 0.0

def compute_behavioural_multiplier(redrob_signals):
    if not redrob_signals:
        return 1.0
        
    open_to_work = redrob_signals.get("open_to_work_flag", True)
    if not open_to_work:
        return 0.0
        
    mult = 1.0
    
    resp_rate = redrob_signals.get("recruiter_response_rate")
    if resp_rate is not None:
        if resp_rate >= 0.5:
            mult *= 1.2
        elif resp_rate < 0.2:
            mult *= 0.8
            
    last_active = redrob_signals.get("last_active_date")
    if last_active:
        try:
            # Handle possible ISO formats
            if last_active.endswith('Z'):
                last_active = last_active[:-1] + '+00:00'
            last_date = datetime.fromisoformat(last_active)
            if last_date.tzinfo is None:
                last_date = last_date.replace(tzinfo=timezone.utc)
            days_since = (datetime.now(timezone.utc) - last_date).days
            if days_since > 90:
                mult *= 0.5
            elif days_since > 30:
                mult *= 0.8
        except Exception:
            pass # Ignore parsing errors
            
    views = redrob_signals.get("profile_views_received_30d")
    if views is not None:
        if views > 100:
            mult *= 1.2
        elif views > 50:
            mult *= 1.1
            
    saved = redrob_signals.get("saved_by_recruiters_30d")
    if saved is not None:
        if saved > 10:
            mult *= 1.1
        elif saved > 5:
            mult *= 1.05
            
    notice = redrob_signals.get("notice_period_days")
    if notice is not None and notice > 60:
        mult *= 0.8
        
    interview_rate = redrob_signals.get("interview_completion_rate")
    if interview_rate is not None and interview_rate < 0.5:
        mult *= 0.9
        
    offer_rate = redrob_signals.get("offer_acceptance_rate")
    if offer_rate is not None and offer_rate >= 0.5:
        mult *= 1.05
        
    return min(2.0, max(0.0, mult))

def compute_location_penalty(profile, redrob_signals):
    loc = str(profile.get("location", "")).lower()
    
    # Check willing_to_relocate in redrob_signals
    willing = redrob_signals.get("willing_to_relocate", False)
    
    if "pune" not in loc and "noida" not in loc:
        if not willing:
            return 0.3
    return 0.0

def detect_honeypot(candidate):
    skills = candidate.get("skills", [])
    profile = candidate.get("profile", {})
    education = candidate.get("education", [])
    
    expert_count = 0
    years_exp = float(profile.get("years_of_experience", 0))
    current_title = str(profile.get("current_title", "")).lower()
    
    # 1. Impossible age/experience anomaly (e.g. over 50 years of experience)
    if years_exp > 50:
        return True
        
    # 2. Graduation year versus claimed experience inconsistency
    import re
    for edu in education:
        end_date = str(edu.get("end_date", ""))
        match = re.search(r'\b(20\d{2})\b', end_date)
        if match:
            grad_year = int(match.group(1))
            # Assuming max overlap allowed is 4 years. If Grad + Exp > Current Year + 4 -> fake
            if (grad_year + years_exp) > 2030: 
                return True
                
    # 3. Unrealistic title seniority compared to experience
    senior_titles = ["director", "vp", "chief", "cto", "head"]
    if any(st in current_title for st in senior_titles) and years_exp < 4:
        return True
        
    # Tool release lifespans (max possible duration in months approx as of 2026)
    tool_lifespans = {
        "langchain": 48,      # Released late 2022 (approx 4 years ago)
        "pytorch": 120,       # Released 2016
        "tensorflow": 132,    # Released 2015
        "huggingface": 96,    # Transformers released ~2018
        "fastapi": 96,        # Released ~2018
        "react": 156          # Released ~2013
    }
    
    for skill in skills:
        name = str(skill.get("name", "")).lower()
        prof = str(skill.get("proficiency", "")).lower()
        duration = float(skill.get("duration_months", 0))
        
        # 4. Expert proficiency with virtually no experience
        if prof == "expert":
            expert_count += 1
            if duration < 6:
                return True
                
        # 5. Temporal paradox (Claiming more experience than the tool has existed)
        for tool, max_months in tool_lifespans.items():
            if tool in name and duration > max_months:
                return True
                
    # 6. Inconsistent expert count vs overall experience
    if expert_count > 3 and years_exp < 2:
        return True
    if expert_count > 5 and years_exp < 5:
        return True
        
    return False

def compute_ats_score(candidate):
    profile = candidate.get("profile", {})
    career_history = candidate.get("career_history", [])
    skills = candidate.get("skills", [])
    
    # Text collection
    summary = str(profile.get("summary", "")).lower()
    exp_text = " ".join([str(job.get("description", "")).lower() for job in career_history])
    full_text = summary + " " + exp_text
    
    # 1. Keyword Match Score (35%)
    kw_matched = sum(1 for kw in ATS_JD_KEYWORDS if kw in full_text or any(kw in str(s.get("name", "")).lower() for s in skills))
    kw_score = (kw_matched / len(ATS_JD_KEYWORDS)) * 100 if ATS_JD_KEYWORDS else 0
    
    # 2. Role Relevance Score (30%)
    titles = [str(profile.get("current_title", "")).lower()]
    titles.extend([str(job.get("title", "")).lower() for job in career_history])
    
    role_score = 0
    for title in titles:
        if any(r in title for r in ATS_ROLES):
            role_score = 100
            break
        elif "engineer" in title and ("ai" in title or "ml" in title or "machine learning" in title):
            role_score = max(role_score, 80)
            
    if role_score == 0:
        if any("data" in t or "software" in t for t in titles):
            role_score = 40
            
    # 3. Skill Density Score (20%)
    total_skills = len(skills)
    if total_skills == 0:
        skill_density = 0
    else:
        relevant_skills = 0
        all_target_skills = set(ATS_JD_KEYWORDS + PROD_ML_KEYWORDS + HIGH_VALUE_SKILLS)
        for s in skills:
            s_name = str(s.get("name", "")).lower()
            if any(ts in s_name for ts in all_target_skills):
                relevant_skills += 1
        skill_density = (relevant_skills / total_skills) * 100
        
    # 4. Resume Quality Score (15%)
    action_words_found = sum(full_text.count(word) for word in ATS_STRONG_TERMS)
    # Cap at 8 strong words for a perfect 100 score
    quality_score = min(100, (action_words_found / 8) * 100)
    
    final_ats = (0.35 * kw_score) + (0.30 * role_score) + (0.20 * skill_density) + (0.15 * quality_score)
    
    return {
        "ats_score": final_ats,
        "ats_keyword": kw_score,
        "ats_role": role_score,
        "ats_density": skill_density,
        "ats_quality": quality_score
    }

def compute_recruiter_score(candidate):
    profile = candidate.get("profile", {})
    career_history = candidate.get("career_history", [])
    education = candidate.get("education", [])
    
    # 1. Profile Relevance Score (30%)
    curr_title = str(profile.get("current_title", "")).lower()
    years_exp = float(profile.get("years_of_experience", 0))
    
    title_score = 0
    if any(r in curr_title for r in ATS_ROLES):
        title_score = 100
    elif "engineer" in curr_title or "scientist" in curr_title or "data" in curr_title:
        title_score = 50
        
    exp_score = 100 if years_exp >= 2.0 else (years_exp / 2.0) * 100
    
    edu_score = 0
    edu_text = " ".join([str(e.get("degree", "")).lower() + " " + str(e.get("major", "")).lower() for e in education])
    if any(t in edu_text for t in REC_EDU_TERMS):
        edu_score = 100
        
    profile_relevance = (title_score * 0.5) + (exp_score * 0.3) + (edu_score * 0.2)
    
    # 2. Experience Relevance Score (30%)
    exp_text = " ".join([str(job.get("description", "")).lower() + " " + str(job.get("title", "")).lower() for job in career_history])
    high_value_words = ["retrieval", "recommendation", "search", "ranking", "nlp", "machine learning", "llm", "production"]
    matched_hv = sum(1 for hw in high_value_words if hw in exp_text)
    exp_relevance = min(100, (matched_hv / 4) * 100) # Cap at 4 matches
    
    # 3. Evidence Score (25%)
    evidence_matches = sum(exp_text.count(t) for t in REC_EVIDENCE_TERMS)
    weak_matches = sum(exp_text.count(t) for t in REC_WEAK_TERMS)
    
    evidence_score = (evidence_matches * 10) - (weak_matches * 10)
    evidence_score = max(0, min(100, evidence_score))
    
    # 4. Career Consistency Score (10%)
    consistency_score = 100
    if len(career_history) > 0:
        total_months = 0
        for job in career_history:
            total_months += float(job.get("duration_months", 0))
        avg_tenure = total_months / len(career_history)
        
        if years_exp > 3 and avg_tenure < 8:
            consistency_score -= 50
            
        # Detect chaotic switching
        job_domains = []
        for job in career_history:
            t = str(job.get("title", "")).lower()
            if "sales" in t or "hr" in t or "marketing" in t or "support" in t:
                job_domains.append("non-tech")
            elif "engineer" in t or "developer" in t or "data" in t or "ai" in t or "ml" in t:
                job_domains.append("tech")
        
        if "non-tech" in job_domains and "tech" in job_domains:
            consistency_score -= 30
            
    consistency_score = max(0, consistency_score)
    
    # 5. Company Credibility Bonus (5%)
    credibility_bonus = 0
    all_companies = " ".join([str(job.get("company", "")).lower() for job in career_history])
    if any(tc in all_companies for tc in REC_TOP_COMPANIES):
        credibility_bonus = 5
        
    final_recruiter = (0.30 * profile_relevance) + (0.30 * exp_relevance) + (0.25 * evidence_score) + (0.10 * consistency_score) + credibility_bonus
    final_recruiter = min(100, final_recruiter)
    
    return {
        "recruiter_score": final_recruiter,
        "rec_profile": profile_relevance,
        "rec_experience": exp_relevance,
        "rec_evidence": evidence_score,
        "rec_consistency": consistency_score,
        "rec_credibility": credibility_bonus
    }

def process_candidate(candidate_json):
    try:
        candidate = json.loads(candidate_json)
    except Exception:
        return None
        
    cid = candidate.get("candidate_id")
    if not cid:
        return None
        
    profile = candidate.get("profile", {})
    career_history = candidate.get("career_history", [])
    skills = candidate.get("skills", [])
    redrob_signals = candidate.get("redrob_signals", {})
    
    title_rel, title_match_kw = extract_title_relevance(profile, career_history)
    prod_ml, prod_ml_yrs = extract_prod_ml_experience(career_history)
    skill_rel, skill_summary = extract_skill_relevance(skills)
    
    consulting_pen = is_consulting_only(career_history)
    research_pen = is_research_only(career_history, prod_ml)
    hopping_pen = compute_title_hopping_penalty(career_history)
    langchain_pen = compute_langchain_only_penalty(skills, skill_rel)
    loc_pen = compute_location_penalty(profile, redrob_signals)
    
    beh_mult = compute_behavioural_multiplier(redrob_signals)
    honeypot = detect_honeypot(candidate)
    
    ats_metrics = compute_ats_score(candidate)
    recruiter_metrics = compute_recruiter_score(candidate)
    
    # Calculate an initial raw score (can be adjusted downstream)
    base_score = (title_rel + prod_ml + skill_rel) / 3.0
    penalties = consulting_pen + research_pen + hopping_pen + langchain_pen + loc_pen
    raw_score = max(0.0, base_score - penalties) * beh_mult
    
    result = {
        "candidate_id": cid,
        "title_relevance": title_rel,
        "prod_ml_experience_score": prod_ml,
        "skill_relevance_score": skill_rel,
        "consulting_only_penalty": consulting_pen,
        "research_only_penalty": research_pen,
        "title_hopping_penalty": hopping_pen,
        "langchain_only_penalty": langchain_pen,
        "behavioural_multiplier": beh_mult,
        "location_penalty": loc_pen,
        "honeypot_flag": honeypot,
        "raw_score": raw_score,
        "title_match_keywords": title_match_kw,
        "prod_ml_years": prod_ml_yrs,
        "skill_set_summary": skill_summary
    }
    
    result.update(ats_metrics)
    result.update(recruiter_metrics)
    return result

def main():
    parser = argparse.ArgumentParser(description="Process candidate JSONL to Parquet features.")
    parser.add_argument("--input", default="candidates.jsonl", help="Input JSONL file (can be gzipped)")
    parser.add_argument("--output", default="candidates_features.parquet", help="Output Parquet file")
    parser.add_argument("--sample", type=int, default=None, help="Process only the first N candidates for a smaller file")
    args = parser.parse_args()
    
    input_file = args.input
    output_file = args.output
    
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} not found.")
        return

    features = []
    
    open_func = gzip.open if input_file.endswith('.gz') else open
    
    print(f"Processing {input_file}...")
    with open_func(input_file, 'rt', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if args.sample and i >= args.sample:
                break
                
            if not line.strip():
                continue
            
            res = process_candidate(line)
            if res:
                features.append(res)
                
            if (i + 1) % 10000 == 0:
                print(f"Processed {i + 1} candidates...")
                
    print(f"Finished processing. Total parsed candidates: {len(features)}")
    
    if features:
        df = pd.DataFrame(features)
        df.to_parquet(output_file, engine='pyarrow', index=False)
        print(f"Successfully saved features to {output_file}")
        
        # Save metadata
        metadata = {
            "num_candidates": len(features),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "columns": list(df.columns)
        }
        with open("metadata.json", "w", encoding="utf-8") as mf:
            json.dump(metadata, mf, indent=2)
            
if __name__ == "__main__":
    main()
