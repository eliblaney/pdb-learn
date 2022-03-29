cargo rustc --lib --release -- -C link-arg=-undefined -C link-arg=dynamic_lookup
mv target/release/librpdb.dylib ../rpdb.so
