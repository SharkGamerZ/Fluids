# /lib/imgui/CMakeLists.txt

# Aggiungi i file di ImGui al tuo obiettivo (ad esempio, una libreria condivisa)
add_library(imgui STATIC
    imgui.cpp
    imstb_rectpack.h
    imstb_textedit.h
    imstb_truetype.h
    imgui_tables.cpp
    imgui_demo.cpp
    imgui_draw.cpp
    imgui_widgets.cpp
    imgui_internal.h
    imgui_impl_glfw.h
    imgui_impl_glfw.cpp
    imgui_impl_opengl3.h
    imgui_impl_opengl3.cpp
    imgui_impl_opengl3_loader.h

)


# Specifica le include directory per ImGui
target_include_directories(imgui PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})