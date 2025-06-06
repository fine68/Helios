import os
pdf_dir = "/openbayes/input/input0/WOS/integrated energy_papers"
output_dir = "./WOS_markdown/integrated energy_papers"  
command = f'marker "{pdf_dir}" --output_dir "{output_dir}" --output_format markdown --disable_image_extraction --workers 5'
os.system(command)