from PIL import Image, ImageSequence

gif1 = Image.open("left_walk.gif")
gif2 = Image.open("right_walk.gif")

frames = [frame.copy() for frame in ImageSequence.Iterator(gif1)] + [
    frame.copy() for frame in ImageSequence.Iterator(gif2)
]

frames[0].save(
    "combined_walk.gif",
    save_all=True,
    append_images=frames[1:],
    duration=gif1.info.get("duration", 100),
    loop=0,
    disposal=2,
)