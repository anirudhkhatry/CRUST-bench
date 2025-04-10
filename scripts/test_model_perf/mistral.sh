# python run.py --benchmark_dir $1 --output_dir $2/claude --prompt /mnt/nas/anirudh/C_to_Rust/scripts/prompts/force_gpt/transpilation_prompts/bullet_point/bullet_point.prompt --prompt_format bullet_point_with_system_instructions --prompt_strategy all --repairer_prompt /mnt/nas/anirudh/C_to_Rust/scripts/prompts/safe_rust_ffi/repair_prompts/bullet_point/bullet_point.prompt --repairer_format bullet_point_with_system_instructions --repairer_strategy all --iterations 5 --mode normal --endpoint claude

cd /mnt/nas/anirudh/C_to_Rust/scripts
cdir='/mnt/nas/anirudh/exps/CBench'
rdir='/mnt/nas/anirudh/exps/RustBench'
odir='/mnt/nas/anirudh/exps/Results'
model="mixtral"
for config in "test_seg" "test_normal"
do
        
    python run.py --benchmark_dir "$cdir" --output_dir "$odir/$model/$config/bullet_point_with_system_instructions" --prompt /mnt/nas/anirudh/C_to_Rust/scripts/prompts/safe_rust_ffi/transpilation_prompts/bullet_point/bullet_point_interface.prompt --prompt_format bullet_point_with_system_instructions --prompt_strategy all --repairer_prompt /mnt/nas/anirudh/C_to_Rust/scripts/prompts/safe_rust_ffi/repair_prompts/bullet_point/bullet_point.prompt --repairer_format bullet_point_with_system_instructions --repairer_strategy all --iterations 5 --mode $config --endpoint "$model" --rust_dir "$rdir" --config /mnt/nas/anirudh/C_to_Rust/scripts/endpoints/configs/mixtral_large.json
    
    python run.py --benchmark_dir "$cdir" --output_dir "$odir/$model/$config/markdown_with_system_instructions" --prompt /mnt/nas/anirudh/C_to_Rust/scripts/prompts/safe_rust_ffi/transpilation_prompts/markdown/safe_rust_ffi_system_interface.prompt --prompt_format markdown_with_system_instructions --prompt_strategy all --repairer_prompt /mnt/nas/anirudh/C_to_Rust/scripts/prompts/safe_rust_ffi/repair_prompts/markdown/safe_rust_ffi_repair_system.prompt --repairer_format markdown_with_system_instructions --repairer_strategy all --iterations 5 --mode $config --endpoint "$model" --rust_dir "$rdir" --config /mnt/nas/anirudh/C_to_Rust/scripts/endpoints/configs/mixtral_large.json


done

