import gradio as gr
import nebula as nb

nf = nb.NebulaFunc()

def sentiment_analysis(text):
    if not text.strip():
        return "Please enter how you're feeling to get a sentiment analysis."
    try:
        result = nf.sentimentAn(text)
        return result
    except Exception as e:
        return f"Error: {str(e)}"




interface = gr.Interface(
    fn=sentiment_analysis,
    inputs=gr.Textbox(
        label="How are you feeling?",
        placeholder="Type how you're feeling...",
        elem_classes=["custom-input"],
        lines=3
    ),
    outputs=gr.Textbox(
        label="Sentiment Analysis Result",
        elem_classes=["custom-output"],
        lines=3
    ),
    title="<span style='color: #FF1493; font-size: 3rem; font-family: Arial, sans-serif;'>Nebula - Your Sentiment Analysis Assistant</span>",


    description="""<div style="text-align: center; margin-bottom: 2rem;">
        <h1 style="color: #FF1493; font-family: 'Arial', sans-serif; font-size: 2.5rem; margin-bottom: 1rem;">Welcome to Nebula!</h1>
        <p style="color: #FF1493; font-size: 1.2rem; line-height: 1.5;">Your sentiment analysis assistant , detects and understands emotions—happy or sad—with precision! 💡💬</p>
    </div>""",

    examples=[
        "I'm feeling great!",
        "Not my best day.",
        "Damn, I'm so happy"
    ],
    theme=gr.themes.Default(
        primary_hue="pink",
        secondary_hue="pink",
        neutral_hue="slate",
        text_size="lg",
        spacing_size="lg",
        radius_size="lg"
    ).set(
        body_background_fill_dark="#000000",

        button_primary_background_fill="#000000",
        button_primary_text_color_dark="#FFFFFF",
        button_primary_background_fill_hover="#333333",
        input_background_fill_dark="#000000",
        input_border_color_dark="#FFFFFF",
        input_border_width="2px",
        input_padding="1rem",
        input_radius="lg",
        button_border_width="2px"

    ),

    allow_flagging="never"
)

if __name__ == "__main__":
    try:
        interface.launch(share=True, server_port=7862, quiet=True)
    except KeyboardInterrupt:
        print("Server shutdown gracefully")
    except Exception as e:
        print(f"Error: {str(e)}")
