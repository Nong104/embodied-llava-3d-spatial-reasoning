@echo off
python -m scripts.train_single_strategy --strategy early --steps 300 --seed 42 --batch-size 2
python -m scripts.train_single_strategy --strategy early --steps 300 --seed 123 --batch-size 2
python -m scripts.train_single_strategy --strategy early --steps 300 --seed 2024 --batch-size 2
python -m scripts.train_single_strategy --strategy late --steps 300 --seed 42 --batch-size 2
python -m scripts.train_single_strategy --strategy late --steps 300 --seed 123 --batch-size 2
python -m scripts.train_single_strategy --strategy late --steps 300 --seed 2024 --batch-size 2
python -m scripts.train_single_strategy --strategy moe --steps 300 --seed 42 --batch-size 2
python -m scripts.train_single_strategy --strategy moe --steps 300 --seed 123 --batch-size 2
python -m scripts.train_single_strategy --strategy moe --steps 300 --seed 2024 --batch-size 2
python -m scripts.train_single_strategy --strategy qformer --steps 300 --seed 42 --batch-size 2
python -m scripts.train_single_strategy --strategy qformer --steps 300 --seed 123 --batch-size 2
python -m scripts.train_single_strategy --strategy qformer --steps 300 --seed 2024 --batch-size 2
echo ALL DONE
pause