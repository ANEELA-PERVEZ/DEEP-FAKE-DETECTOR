import gradio as gr
import os

# --- 1. IMPORT FUNCTIONS FROM YOUR CLEANED FILES ---
from fake_text import hybrid_analysis
from fake_audio import analyze_voice_forensics
from fake_video import analyze_video as video_pipeline
from final_lipsync import master_pipeline as lipsync_pipeline
from fake_image import predict_image as image_pipeline

# --- 2. BUILD THE UNIFIED DASHBOARD ---
with gr.Blocks(theme=gr.themes.Soft(), title="Deepfake Detector") as demo:
    gr.Markdown("<h1 style='text-align: center;'>🛡️ Deepfake Detection System</h1>")
    gr.Markdown("<p style='text-align: center;'>Comprehensive analysis across Text, Audio, Video, and Lip-Sync domains.</p>")
    
    with gr.Tabs():
        
        # --- TAB 1: FAKE NEWS (TEXT) ---
        with gr.TabItem("📰 Fake Text/News"):
            gr.Markdown("### 🧠 Text Claim Analysis")
            with gr.Row():
                text_in = gr.Textbox(label="Enter News Claim", placeholder="e.g., The Eiffel Tower was sold today...")
            with gr.Row():
                text_btn = gr.Button("🔍 Analyze Claim", variant="primary")
            with gr.Row():
                text_out = gr.HTML(label="Verdict & Reasoning")
            
            text_btn.click(hybrid_analysis, inputs=[text_in], outputs=[text_out])

        # --- TAB 2: FAKE AUDIO ---
        with gr.TabItem("🎙️ Audio Forensics"):
            gr.Markdown("### 🔊 Audio Texture & Splice Analysis")
            with gr.Row():
                aud_in = gr.Audio(type="filepath", label="Upload audio file")
            with gr.Row():
                aud_btn = gr.Button("🔍 Run", variant="primary")
            with gr.Row():
                with gr.Column():
                    aud_v_out = gr.Textbox(label="Final Remarks")
                    aud_e_out = gr.Textbox(label="Categorized Evidence (Metrics)")
                with gr.Column():
                    aud_r_out = gr.Textbox(label="Reasoning", lines=4)
            
            aud_btn.click(analyze_voice_forensics, inputs=[aud_in], outputs=[aud_v_out, aud_e_out, aud_r_out])

        # --- TAB 3: FAKE VIDEO ---
        with gr.TabItem("🎥 Video Forensics"):
            gr.Markdown("### 🏃 Video analysis")
            with gr.Row():
                vid_in = gr.Video(label="Upload Video for Analysis")
                vid_btn = gr.Button("🔍 Analyze Video", variant="primary")
            with gr.Row():
                with gr.Column():
                    vid_img_out = gr.Image(label="Suspicious Keyframe", type="pil")
                with gr.Column():
                    vid_txt_out = gr.Markdown()
            with gr.Row():
                vid_reason_out = gr.Textbox(label="Reasoning", lines=5)
                vid_hidden = gr.Textbox(visible=False) 
            
            vid_btn.click(video_pipeline, inputs=[vid_in], outputs=[vid_img_out, vid_txt_out, vid_reason_out, vid_hidden])

        # --- TAB 4: Multi-Modal Analysis (LIP-SYNC) ---
        with gr.TabItem("👄 Multi-Modal Analysis (Lip-Sync)"):
            gr.Markdown("### ⚖️ Multi-Modal Analysis: Video + Audio")
            with gr.Row():
                master_vid_in = gr.Video(label="Upload Video")
                master_btn = gr.Button("🔍 Run Analysis", variant="primary")
            with gr.Row():
                with gr.Column():
                    master_img_out = gr.Image(label="Keyframe Analysis")
                with gr.Column():
                    master_txt_out = gr.Markdown(label=" Report")
            with gr.Row():
                master_reason_out = gr.Textbox(label=" Reasoning", lines=3)
            
            master_btn.click(lipsync_pipeline, inputs=[master_vid_in], outputs=[master_img_out, master_txt_out, master_reason_out])

        # --- TAB 5: IMAGE DETECTION ---
        with gr.Tab("🖼️ Image Analysis"):
            gr.Markdown(" Deepfake Images Detector")
            with gr.Row():
                with gr.Column():
                    img_in = gr.Image(type="pil", label=" Upload Image")
                    img_btn = gr.Button("Analyze Image", variant="primary")
                with gr.Column():
                    img_out = gr.Label(num_top_classes=2, label="Detection Result")
        
            img_btn.click(fn=image_pipeline, inputs=[img_in], outputs=[img_out])

# --- 3. LAUNCH THE APP ---
if __name__ == "__main__":
    demo.launch(share=True, debug=True)