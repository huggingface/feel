import openai
import os 
from alpaca_eval import run_evaluation

def judge_responses(response1, response2, prompt):
    """
    Use OpenAI GPT-4 API to judge two model responses.
    Returns: "A" if response1 is better, "B" if response2 is better, or "tie".
    """
    openai.api_key = os.getenv("OPENAI_API_KEY")

    prompt_text = f"""
    Given the user prompt: "{prompt}"
    
    Response A: "{response1}"
    Response B: "{response2}"
    
    Which response is better? Reply with 'A', 'B', or 'tie'.
    """

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "system", "content": "You are an expert evaluator."},
                      {"role": "user", "content": prompt_text}],
            max_tokens=5
        )
        result = response["choices"][0]["message"]["content"].strip().lower()
        return result if result in ["a", "b", "tie"] else "tie"
    except Exception as e:
        print(f"Error in OpenAI API call: {e}")
        return "tie"



def alpaca_evaluator(model_name, num_samples=200):
    results = run_evaluation(
        model=model_name,
        num_samples=num_samples,  # fewer samples for quick testing
        reference_model="gpt-4",  # Compare against GPT-4 (optional)
    )
    return results 

