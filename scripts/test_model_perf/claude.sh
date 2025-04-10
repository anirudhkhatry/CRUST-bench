
cd ../../src
cdir=$1
rdir=$2
odir=$3

model="claude"
config="test_normal"
    
python run.py \
    --benchmark_dir "$cdir" \
    --output_dir "$odir/$model/$config/bullet_point_with_system_instructions_repair_updated" \
    --prompt ./prompts/transpilation_prompts/bullet_point/bullet_point_interface.prompt \
    --prompt_format bullet_point_with_system_instructions \
    --prompt_strategy all \
    --repairer_prompt ./prompts/repair_prompts/bullet_point/bullet_point.prompt \
    --repairer_format bullet_point_with_system_instructions \
    --repairer_strategy all \
    --iterations 3 \
    --mode $config \
    --endpoint "$model" \
    --rust_dir "$rdir"
python run.py \
    --benchmark_dir "$cdir" \
    --output_dir "$odir/$model/$config/bullet_point_with_system_instructions_single_updated" \
    --prompt ./prompts/transpilation_prompts/bullet_point/bullet_point_interface.prompt \
    --prompt_format bullet_point_with_system_instructions \
    --prompt_strategy all \
    --repairer_prompt ./prompts/repair_prompts/bullet_point/bullet_point.prompt \
    --repairer_format bullet_point_with_system_instructions \
    --repairer_strategy all \
    --iterations 0 \
    --mode $config \
    --endpoint "$model" \
    --rust_dir "$rdir"