import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from ranking_engine import rank_candidates, DEFAULT_WEIGHTS

st.set_page_config(page_title="Candidate Ranking Dashboard", layout="wide")

@st.cache_data
def load_features():
    try:
        return pd.read_parquet("candidates_features.parquet")
    except Exception as e:
        st.error(f"Error loading features: {e}. Please ensure candidates_features.parquet exists.")
        return pd.DataFrame()

df_features = load_features()

if df_features.empty:
    st.stop()

# --- Sidebar: Weight Sliders ---
st.sidebar.title("Ranking Weights")
st.sidebar.markdown("Adjust the formula weights dynamically.")

if "weights" not in st.session_state or "w_recruiter" not in st.session_state.weights:
    st.session_state.weights = DEFAULT_WEIGHTS.copy()

def reset_weights():
    st.session_state.weights = DEFAULT_WEIGHTS.copy()

st.sidebar.button("Reset to Default", on_click=reset_weights)

current_weights = {}
current_weights["w_ats"] = st.sidebar.slider("ATS Score Weight", 0.0, 1.0, st.session_state.weights.get("w_ats", 0.30), 0.05)
current_weights["w_recruiter"] = st.sidebar.slider("Recruiter Score Weight", 0.0, 1.0, st.session_state.weights.get("w_recruiter", 0.25), 0.05)
current_weights["w_prod_ml"] = st.sidebar.slider("Production ML Weight", 0.0, 1.0, st.session_state.weights.get("w_prod_ml", 0.20), 0.05)
current_weights["w_skill"] = st.sidebar.slider("Skill Relevance Weight", 0.0, 1.0, st.session_state.weights.get("w_skill", 0.15), 0.05)
current_weights["w_behaviour"] = st.sidebar.slider("Behaviour Weight", 0.0, 1.0, st.session_state.weights.get("w_behaviour", 0.10), 0.05)
current_weights["w_penalty"] = st.sidebar.slider("Penalty Weight", 0.0, 1.0, st.session_state.weights.get("w_penalty", 0.10), 0.05)

# Keep state synced
st.session_state.weights = current_weights

# Compute ranking based on current weights
@st.cache_data
def cached_rank_candidates(features_df, weights, top_n=100):
    return rank_candidates(features_df, weights=weights, top_n=top_n)

top_100 = cached_rank_candidates(df_features, current_weights, 100)
# Join full features for detailed view
top_100_full = top_100.merge(df_features, on="candidate_id", how="left")

# --- Export CSV ---
csv_data = top_100[["candidate_id", "rank", "score", "reasoning"]].to_csv(index=False)
st.sidebar.download_button(
    label="Download submission.csv",
    data=csv_data,
    file_name="submission.csv",
    mime="text/csv"
)

# --- Weight Chart ---
st.sidebar.markdown("### Weight Importance")
weight_df = pd.DataFrame(list(current_weights.items()), columns=["Factor", "Weight"])
fig_weights = px.bar(weight_df, x="Weight", y="Factor", orientation='h', height=300)
fig_weights.update_layout(margin=dict(l=0, r=0, t=0, b=0))
st.sidebar.plotly_chart(fig_weights, use_container_width=True)


# --- Main Area: Tabs ---
st.title("🏆 Redrob Candidate Ranking Dashboard")

tab1, tab2, tab3, tab4 = st.tabs(["Top 100 Ranking", "Compare Candidates", "What-If Simulator", "Methodology"])

with tab1:
    st.header("Top 100 Candidates")
    st.dataframe(
        top_100[["rank", "candidate_id", "score", "reasoning"]], 
        use_container_width=True,
        hide_index=True
    )
    
    st.markdown("### Candidate Details")
    selected_cid = st.selectbox("Select a candidate to view full details", top_100["candidate_id"].tolist())
    
    if selected_cid:
        cand_data = top_100_full[top_100_full["candidate_id"] == selected_cid].iloc[0]
        
        st.subheader(f"Details for {selected_cid} (Rank {cand_data['rank']})")
        st.markdown(f"**Reasoning:** {cand_data['reasoning']}")
        
        st.markdown("---")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.markdown("**Technical Features**")
            st.write(f"Prod ML Score: {cand_data.get('prod_ml_experience_score', 0):.2f} ({cand_data.get('prod_ml_years', 0):.1f} yrs)")
            st.write(f"Skill Relevance: {cand_data.get('skill_relevance_score', 0):.2f}")
            st.write(f"Raw Title Keywords: {cand_data.get('title_match_keywords', '')}")
            st.write(f"Raw Skills: {cand_data.get('skill_set_summary', '')}")
            
        with col2:
            st.markdown("**ATS Simulator**")
            st.write(f"**Total ATS:** {cand_data.get('ats_score', 0):.1f}/100")
            st.write(f"Keyword Match: {cand_data.get('ats_keyword', 0):.1f}")
            st.write(f"Role Relevance: {cand_data.get('ats_role', 0):.1f}")
            st.write(f"Skill Density: {cand_data.get('ats_density', 0):.1f}")
            st.write(f"Resume Quality: {cand_data.get('ats_quality', 0):.1f}")
            
        with col3:
            st.markdown("**Recruiter Review**")
            st.write(f"**Total Recruiter:** {cand_data.get('recruiter_score', 0):.1f}/100")
            st.write(f"Profile: {cand_data.get('rec_profile', 0):.1f}")
            st.write(f"Experience: {cand_data.get('rec_experience', 0):.1f}")
            st.write(f"Evidence: {cand_data.get('rec_evidence', 0):.1f}")
            st.write(f"Consistency: {cand_data.get('rec_consistency', 0):.1f}")
            st.write(f"Credibility Bonus: +{cand_data.get('rec_credibility', 0)}")
            
        with col4:
            st.markdown("**Penalties**")
            st.write(f"Consulting Penalty: {cand_data.get('consulting_only_penalty', 0):.2f}")
            st.write(f"Research Penalty: {cand_data.get('research_only_penalty', 0):.2f}")
            st.write(f"Hopping Penalty: {cand_data.get('title_hopping_penalty', 0):.2f}")
            st.write(f"LangChain Penalty: {cand_data.get('langchain_only_penalty', 0):.2f}")
            st.write(f"Location Penalty: {cand_data.get('location_penalty', 0):.2f}")
            
        with col5:
            st.markdown("**Behaviour & Flags**")
            st.write(f"Behavioural Multiplier: {cand_data.get('behavioural_multiplier', 1):.2f}")
            st.write(f"Honeypot Flag: {cand_data.get('honeypot_flag', False)}")

with tab2:
    st.header("Compare Candidates")
    colA, colB = st.columns(2)
    cand_list = top_100["candidate_id"].tolist()
    
    with colA:
        cand_a = st.selectbox("Select Candidate A", cand_list, index=0)
    with colB:
        cand_b = st.selectbox("Select Candidate B", cand_list, index=1 if len(cand_list) > 1 else 0)
        
    if cand_a and cand_b:
        data_a = top_100_full[top_100_full["candidate_id"] == cand_a].iloc[0]
        data_b = top_100_full[top_100_full["candidate_id"] == cand_b].iloc[0]
        
        metrics = [
            "ats_score", "recruiter_score", "prod_ml_experience_score", "skill_relevance_score",
            "consulting_only_penalty", "research_only_penalty", "title_hopping_penalty",
            "behavioural_multiplier", "score"
        ]
        
        comp_df = pd.DataFrame({
            "Metric": metrics,
            cand_a: [data_a.get(m, 0) for m in metrics],
            cand_b: [data_b.get(m, 0) for m in metrics]
        })
        
        st.dataframe(comp_df, use_container_width=True, hide_index=True)
        
        # Spider/Radar Chart
        fig = go.Figure()
        
        # Helper to normalize for radar chart
        def norm_val(val, metric):
            if metric in ["ats_score", "recruiter_score"]:
                return val / 100.0
            return val
            
        fig.add_trace(go.Scatterpolar(
            r=[norm_val(data_a.get(m, 0), m) for m in metrics[:4]] + [norm_val(data_a.get(metrics[0], 0), metrics[0])],
            theta=["ATS", "Recruiter", "Prod ML", "Skills", "ATS"],
            fill='toself',
            name=cand_a
        ))
        fig.add_trace(go.Scatterpolar(
            r=[norm_val(data_b.get(m, 0), m) for m in metrics[:4]] + [norm_val(data_b.get(metrics[0], 0), metrics[0])],
            theta=["ATS", "Recruiter", "Prod ML", "Skills", "ATS"],
            fill='toself',
            name=cand_b
        ))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.header("What-If Simulator")
    st.markdown("Modify a candidate's features and see how their rank changes globally.")
    
    sim_cand = st.selectbox("Select Candidate for Simulation", top_100["candidate_id"].tolist())
    
    if sim_cand:
        orig_data = df_features[df_features["candidate_id"] == sim_cand].iloc[0].copy()
        orig_rank = top_100_full[top_100_full["candidate_id"] == sim_cand].iloc[0]["rank"]
        orig_score = top_100_full[top_100_full["candidate_id"] == sim_cand].iloc[0]["score"]
        
        st.subheader("Simulation Controls")
        new_beh = st.slider("Behavioural Multiplier", 0.0, 2.0, float(orig_data["behavioural_multiplier"]), 0.05)
        
        col1, col2 = st.columns(2)
        with col1:
            rm_consult = st.checkbox("Remove Consulting Penalty", value=False)
            rm_research = st.checkbox("Remove Research Penalty", value=False)
        with col2:
            rm_hopping = st.checkbox("Remove Hopping Penalty", value=False)
            rm_location = st.checkbox("Remove Location Penalty", value=False)
            
        # Apply simulations
        sim_df = df_features.copy()
        idx = sim_df.index[sim_df["candidate_id"] == sim_cand][0]
        
        sim_df.at[idx, "behavioural_multiplier"] = new_beh
        if rm_consult: sim_df.at[idx, "consulting_only_penalty"] = 0.0
        if rm_research: sim_df.at[idx, "research_only_penalty"] = 0.0
        if rm_hopping: sim_df.at[idx, "title_hopping_penalty"] = 0.0
        if rm_location: sim_df.at[idx, "location_penalty"] = 0.0
        
        # Re-rank with simulation data
        sim_top_100 = rank_candidates(sim_df, weights=current_weights, top_n=100)
        
        # Find new rank
        new_row = sim_top_100[sim_top_100["candidate_id"] == sim_cand]
        if not new_row.empty:
            new_rank = new_row.iloc[0]["rank"]
            new_score = new_row.iloc[0]["score"]
            st.success(f"**New Rank:** {new_rank} (Original: {orig_rank}) | **New Score:** {new_score:.4f} (Original: {orig_score:.4f})")
        else:
            # We need to find their rank beyond Top 100
            full_ranked = rank_candidates(sim_df, weights=current_weights, top_n=len(sim_df))
            new_rank = full_ranked[full_ranked["candidate_id"] == sim_cand].iloc[0]["rank"]
            new_score = full_ranked[full_ranked["candidate_id"] == sim_cand].iloc[0]["score"]
            st.warning(f"**New Rank:** {new_rank} (Original: {orig_rank}) | **New Score:** {new_score:.4f} (Original: {orig_score:.4f})")
            st.info("Candidate dropped out of the Top 100.")

with tab4:
    st.header("Methodology")
    st.markdown("""
    ### Scoring Formula
    The final score is computed as:
    ```
    Final Score = 
        (w_ats * (ATS Score / 100)) + 
        (w_recruiter * (Recruiter Score / 100)) + 
        (w_prod_ml * Production ML) + 
        (w_skill * Skill Relevance) + 
        (w_behaviour * Behaviour Score) - 
        (w_penalty * Total Penalties)
    ```
    
    ### Key Features
    - **Production ML Score**: Highlights real-world deployment experience with vector databases and retrieval systems.
    - **Skill Relevance**: Weights skills by their proficiency, endorsement levels, and duration.
    - **Behavioural Multiplier**: Amplifies candidates who are highly responsive, active recently, and have high offer acceptance rates.
    - **Honeypot Detection**: Zeroes out scores for candidates exhibiting anomalous expert skills relative to their total experience.
    """)
