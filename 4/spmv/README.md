## How To use

1. install rust using the following command

```bash 
curl --proto '=https' --tlsv1.2 https://sh.rustup.rs -sSf | sh
```

1. fill the todos
```bash 
grep todo ./ -r --exclude README.md
```

1. to test your code run the following command
```bash 
cargo test
```

1. to test each module separately , use the following command
```bash 
# cargo test <module_name>
cargo test dens
cargo test single_thread
cargo test multi_thread
cargo test simd
```

1. to run the benchmarks run `cargo bench`

1. to run each benchmark separately, run:
```bash 
# cargo bench --bench <module_name>
cargo bench --bench dense 
cargo bench --bench single_thread
cargo bench --bench multi_thread 
cargo bench --bench simd 
```
