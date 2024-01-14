# /lib/imgui/CMakeLists.txt

# Aggiungi i file di ImGui al tuo obiettivo (ad esempio, una libreria condivisa)
add_library(imgui STATIC
    imgui.cpp
    imgui_demo.cpp
    imgui_draw.cpp
    imgui_widgets.cpp
    imgui_impl_glfw.h
    imgui_impl_glfw.cpp
    imgui_impl_opengl3.h
    
)
    
    # ... Aggiungi altri file se necessario ...


# Specifica le include directory per ImGui
target_include_directories(imgui PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})