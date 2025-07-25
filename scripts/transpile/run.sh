cd ../../src
echo "Please select the model you want to run. Select the name in the brackets."
echo "1. Claude 3.5 (claude)"
echo "2. Claude 3.7 (claude37)"
echo "3. GPT-4 (gpt-4o)"
echo "4. OpenAI o1 (o1)"
echo "5. OpenAI o1-mini (o1-mini)"
echo "6. Gemini 1.5 Pro (gemini)"
echo "Please ensure for the following models you have VLLM running in the background."
echo "7. QwQ 32B (qwq)"
echo "8. Virtuoso 32B (virtuoso)"
echo "9. LLama 3 8b (llama)"

read -p "Enter your choice (please provide the name in brackets): " model
read -p "Enter the config file (e.g., ./endpoints/configs/claude_self_repair.json): " config
read -p "Enter the iterations (e.g., 5): " iterations

c_dir="../datasets/CBench"
odir="../results/$model/bullet_point_with_system_instructions"
rdir="../datasets/RBench"
# Check if the model is valid
if [[ "$model" != "claude" && "$model" != "claude37" && "$model" != "gpt-4o" && "$model" != "o1" && "$model" != "o1-mini" && "$model" != "gemini" && "$model" != "qwq" && "$model" != "virtuoso" && "$model" != "llama" ]]; then
    echo "Invalid model selected. Please run the script again and select a valid model."
    exit 1
fi

python run.py \
    --mode "normal" \
    --benchmark_dir "$c_dir" \
    --output_dir "$odir/$model/$config/bullet_point_with_system_instructions" \
    --prompt ./prompts/transpilation_prompts/bullet_point/bullet_point_interface.prompt \
    --prompt_format bullet_point_with_system_instructions \
    --prompt_strategy all \
    --repairer_prompt ./prompts/repair_prompts/bullet_point/bullet_point.prompt \
    --repairer_format bullet_point_with_system_instructions \
    --repairer_strategy all \
    --iterations "$iterations" \
    --config "$config" \
    --endpoint "$model" \
    --rust_dir "$rdir"