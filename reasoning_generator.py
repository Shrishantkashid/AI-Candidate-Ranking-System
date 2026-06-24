import hashlib

def get_deterministic_choice(cid, choices):
    """Pick a deterministic choice based on candidate_id."""
    h = int(hashlib.sha256(str(cid).encode('utf-8')).hexdigest(), 16)
    return choices[h % len(choices)]

def generate_human_reasoning(row):
    """
    Generates a deterministic, highly varied, recruiter-style reasoning paragraph
    based strictly on computed features. Uses conversational language.
    """
    cid = row.get("candidate_id", "unknown")
    
    # 1. Evaluate Core Strengths
    ats = row.get("ats_score", 0.0)
    rec = row.get("recruiter_score", 0.0)
    prod_ml = row.get("prod_ml_experience_score", 0.0)
    skill_rel = row.get("skill_relevance_score", 0.0)
    prod_yrs = row.get("prod_ml_years", 0.0)
    
    strengths = []
    weaknesses = []
    
    # Analyze ATS vs Recruiter (The Signal)
    if ats > 80 and rec > 80:
        strengths.append(get_deterministic_choice(cid + "_ats_rec", [
            "I really like this one. They tick all the boxes for the skills we need and back it up with solid, quantifiable experience.",
            "Looks like a great fit. Their resume hits all the key technical requirements, and they clearly have the real-world track record we're looking for.",
            "Really strong candidate. The technical stack matches perfectly, and their past roles show undeniable proof of impact.",
            "This is a top-tier profile. They have the exact skills we asked for and serious engineering credibility.",
            "Awesome match on paper. Keyword alignment is incredibly high, plus they have the hands-on proof to back it up.",
            "Definitely worth an interview. They nailed the ATS criteria and showed actual qualitative depth in their past roles.",
            "A rare find where the technical buzzwords actually match deep, verifiable experience.",
            "I'd move fast on this one. Perfect alignment on the tech stack and a super consistent track record of execution.",
            "Very impressed by this resume. Hits all our keyword filters and has the project evidence to prove they're legit.",
            "Checks every single technical box, and their past work experience is incredibly solid.",
            "Standout candidate right here. Strong technical foundation mixed with great recruiter heuristic signals.",
            "It’s hard to find a profile this balanced. They have the exact tooling we need and great career consistency.",
            "Big fan of this profile. High signal-to-noise ratio with excellent keyword coverage and verifiable achievements.",
            "This one is an easy yes for a screening call. Fantastic alignment across the board.",
            "Super robust background. They easily pass our technical bar and show strong evidence of shipping real work."
        ]))
    elif ats > 85 and rec < 50:
        weaknesses.append(get_deterministic_choice(cid + "_ats_rec2", [
            "Their resume looks a bit keyword-stuffed. They mention the right tools, but I'm not seeing enough concrete evidence of what they actually built.",
            "While they have the right buzzwords on paper, the lack of real depth or career progression is a bit concerning.",
            "They hit the keyword filters, but when you look closer, the actual application of those skills seems pretty superficial.",
            "Lots of the right tech terms, but honestly, it lacks the qualitative proof of work I want to see.",
            "The ATS scored them high, but I'm skeptical. The resume reads like a laundry list of tools without real impact.",
            "Good keyword density, but the job history is too chaotic to feel confident about their actual depth.",
            "They know what to put on a resume to pass the filters, but I'm not convinced they have the deep engineering chops.",
            "Technically a match, but the lack of concrete achievements makes me think they might just be a tutorial follower.",
            "Passes the technical screen, but fails the recruiter sniff test. Not enough solid evidence of taking ownership.",
            "Seems like they optimized for ATS. I need to see more real-world impact before getting excited."
        ]))
    elif rec > 85 and ats < 50:
        strengths.append(get_deterministic_choice(cid + "_ats_rec3", [
            "They might not have every single keyword we asked for, but their background is super credible and they clearly know how to ship products.",
            "I'd definitely talk to them. The raw tech stack might not be a 100% match, but their engineering pedigree and proof of work is fantastic.",
            "A really solid qualitative fit. They might be missing a couple of buzzwords, but they more than make up for it with real engineering depth.",
            "Strong engineering fundamentals. Even if they don't know our exact stack, they have a track record of figuring things out.",
            "Their experience is top-notch. I don't care if they missed a few keywords; they are clearly a senior builder.",
            "High-quality profile. The specific tools might differ a bit, but the career progression is incredibly consistent and strong.",
            "I really like their background. Missing a couple of minor tech requirements, but the core engineering talent is obvious.",
            "Great proof of work here. They might need to ramp up on one or two tools, but the foundation is excellent.",
            "This is a great example of a strong engineer who just didn't keyword-optimize their resume. Definitely worth a call.",
            "Very solid history of shipping. The ATS score is a bit low, but human review shows they are highly capable."
        ]))
    
    # Analyze Production ML
    if prod_ml > 0.8:
        ml_skills = str(row.get("skill_set_summary", ""))
        skill_str = ""
        if ml_skills and ml_skills.lower() != "nan":
            skill_list = [s.strip() for s in ml_skills.split(",") if s.strip()]
            if skill_list:
                skill_str = f" particularly with {skill_list[0]} and {skill_list[1]}" if len(skill_list) > 1 else f" mostly around {skill_list[0]}"
                
        strengths.append(get_deterministic_choice(cid + "_prodml", [
            f"Plus, they have deep production ML experience ({prod_yrs:.1f} years), showing they actually know how to deploy models{skill_str}.",
            f"What stands out is their hands-on time taking ML systems to production ({prod_yrs:.1f} years){skill_str}. Not just an academic.",
            f"They've clearly been in the trenches shipping ML features to production, which is exactly what we need right now.",
            f"I love that they actually have {prod_yrs:.1f} years of real production ML experience{skill_str}. Very hard to find.",
            f"Their experience taking models from notebook to production ({prod_yrs:.1f} years) is a massive plus for us.",
            f"They aren't just prototyping; they have a solid {prod_yrs:.1f} years of getting ML systems into production{skill_str}.",
            f"Extensive background in operationalizing ML ({prod_yrs:.1f} years). This is someone who knows how to scale systems.",
            f"The fact that they have deployed ML models in real-world scenarios for {prod_yrs:.1f} years makes them a highly attractive candidate.",
            f"Very strong MLOps and production-level experience here ({prod_yrs:.1f} years). They know the whole lifecycle.",
            f"I’m impressed by the {prod_yrs:.1f} years of applied, production-facing machine learning work.",
            f"It’s great to see a candidate who has actually lived with the models they built in production for {prod_yrs:.1f} years.",
            f"They bring a lot of maturity to the table with {prod_yrs:.1f} years of pure production ML deployment.",
            f"A huge advantage is their proven ability to ship and maintain ML systems over the last {prod_yrs:.1f} years.",
            f"Real-world ML engineering is their strong suit. {prod_yrs:.1f} years of production experience is no joke.",
            f"This is a builder who knows how to productionize algorithms, backed by {prod_yrs:.1f} years of solid evidence."
        ]))
    elif prod_ml < 0.2 and skill_rel > 0.6:
        weaknesses.append(get_deterministic_choice(cid + "_prodml2", [
            "They have the right theory, but I'm worried about their lack of actual production ML deployments.",
            "The tech stack looks fine, but they feel more like a prototyper. We really need someone with more production-grade deployment experience.",
            "Seems like a strong academic or researcher, but they might struggle with the engineering side of taking models to prod.",
            "Lots of notebook-level skills, but I don't see the MLOps or deployment experience we need.",
            "Great theoretical grasp, but a bit too light on the actual engineering required to ship ML products.",
            "I'm concerned they haven't seen the ugly side of production ML yet. Very research-heavy.",
            "They know the math and the frameworks, but the resume lacks evidence of putting models into real production environments.",
            "Strong on modeling, weak on engineering. They might need a lot of hand-holding to actually deploy anything.",
            "Seems more like a data scientist than an ML engineer. Lacks the hardcore deployment background.",
            "While they know the algorithms, the absence of real-world production ML experience is a noticeable gap."
        ]))
        
    # 2. Evaluate Penalties & Behavioral Signals
    hopping_pen = row.get("title_hopping_penalty", 0.0)
    consulting_pen = row.get("consulting_only_penalty", 0.0)
    beh_mult = row.get("behavioural_multiplier", 1.0)
    
    if hopping_pen > 0:
        weaknesses.append(get_deterministic_choice(cid + "_hop", [
            "The job hopping is definitely a red flag for me. Lots of short stints make me worry about retention.",
            "I'm a little concerned about their career consistency given how often they jump ship.",
            "Their average tenure is pretty short, which makes me question if they'll stick around to own a long-term project.",
            "A bit too jumpy for my liking. We need someone who can commit for more than a year.",
            "The constant switching between companies makes me wonder if they struggle to settle in.",
            "Career history is a bit fragmented. Not a lot of long-term tenures.",
            "I'd want to dig into why they leave roles so quickly. The hopping is concerning.",
            "They don't seem to stay anywhere long enough to see the long-term consequences of their code.",
            "Frequent job changes are definitely a risk factor we'd need to address.",
            "The resume shows a pattern of short stints, which makes them a risky hire for a core team."
        ]))
        
    if consulting_pen > 0:
        weaknesses.append(get_deterministic_choice(cid + "_cons", [
            "Their background is entirely in consulting/IT services, so they might have a tough time adjusting to a fast-paced product culture.",
            "It's a heavily service-oriented resume. We'd need to vet if they can handle core product engineering.",
            "Lacks true product-first experience since they've spent their whole career jumping between consulting clients.",
            "Almost entirely agency/consulting work. Sometimes that mindset doesn't translate well to in-house product teams.",
            "They've never worked at a core product company, which could mean a steep learning curve for our environment.",
            "The purely services background makes me wonder if they have deep ownership experience over a single product.",
            "Consulting heavy. We'd have to ensure they actually wrote the core code and didn't just do implementations.",
            "A bit too focused on IT services. Product engineering requires a different kind of long-term ownership.",
            "I usually prefer candidates who have lived with a product, rather than just consulting on it.",
            "The lack of in-house product experience is a slight negative for me."
        ]))
        
    if beh_mult > 1.2:
        strengths.append(get_deterministic_choice(cid + "_beh", [
            "They're also super active on the platform and seem highly responsive, so we should move fast.",
            "Engagement metrics look great. They are actively looking and likely ready to interview right away.",
            "Strong behavioral signals here—they've been very active recently, indicating high intent.",
            "Looks like they are highly engaged and open to opportunities. A warm lead for sure.",
            "The platform data shows they are responding quickly, which is a great sign for the hiring pipeline.",
            "Very fresh activity. If we like them, we need to reach out today.",
            "High intent signals across the board. They are definitely in the market right now.",
            "I love seeing this level of responsiveness. Makes the recruiting process so much smoother.",
            "Their engagement score is top tier. Let's get them in the loop immediately.",
            "Given their recent platform activity, I expect they'd be a very fast mover if we offer an interview."
        ]))
    elif beh_mult < 0.8:
        weaknesses.append(get_deterministic_choice(cid + "_beh2", [
            "They look pretty passive right now, or might have a really long notice period.",
            "Engagement on the platform is pretty low, so they might be slow to respond to outreach.",
            "Recent activity is minimal, meaning this might be a tougher candidate to close quickly.",
            "Behavioral data suggests they aren't actively looking, so it might be an uphill battle.",
            "Looks like a passive candidate. We might have to work hard to get their attention.",
            "The low response rates indicate they might be happily employed and not eager to jump.",
            "Not showing much intent on the platform. Could be a slow process to get them engaged.",
            "Notice periods or lack of activity make this a lower-priority outreach for immediate needs.",
            "They haven't been very active lately, so temper expectations on a quick reply.",
            "Behavioral signals are weak. We can reach out, but don't hold your breath."
        ]))
        
    # 3. Synthesize the final reasoning
    if not strengths and not weaknesses:
        return get_deterministic_choice(cid + "_fallback", [
            "Honestly, a pretty average profile. Meets the baseline but doesn't really stand out from the pack.",
            "An okay fit on paper, but I'm not seeing anything that screams 'must hire'.",
            "They meet the minimum bar for the role, but there's no major hook or deep production evidence to push them to the top.",
            "A middle-of-the-road candidate. Good enough for a screen, but probably not our first choice.",
            "Solid, but unspectacular. They check the basic boxes and nothing more.",
            "I'm fairly neutral on this one. Nothing terribly wrong, but nothing overly exciting either.",
            "An acceptable profile. I wouldn't rush to call them, but they are decent enough to keep in the pipeline.",
            "Passable technical skills, but the overall resume lacks a wow factor.",
            "They are perfectly adequate, though I suspect we can find stronger matches.",
            "A standard background. Worth keeping on file if our top choices don't pan out."
        ])
        
    final_text = ""
    if strengths:
        final_text += " ".join(strengths)
    if weaknesses:
        if final_text:
            final_text += " " + get_deterministic_choice(cid + "_trans", [
                "But honestly,", "On the flip side,", "That being said,", 
                "However,", "At the same time,", "Though,", 
                "Then again,", "Keep in mind, though,", 
                "A point of caution, however,", "One thing to note, though,"
            ]) + " "
        final_text += " ".join(weaknesses)
        
    return final_text.strip()
