set(TARGET llama-wf8-s1-gguf-check)
add_executable(${TARGET} wf8-s1-gguf-check.cpp)
install(TARGETS ${TARGET} RUNTIME)
target_link_libraries(${TARGET} PRIVATE ggml ${CMAKE_THREAD_LIBS_INIT})
