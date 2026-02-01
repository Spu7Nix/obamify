[(try it here)](https://obamify.com/)
# obamify
revolutionary new technology that turns any image into obama

![example](example.gif)

# How to use

**Use the ui at the top of the window to control the animation, choose between saved transformations, and generate new ones.** You can change the source image and target image, and choose how they are cropped to a square (tip: if both the images are faces, try making the eyes overlap). You can also change these advanced settings:
| Setting               | Description                                                                                     |
|-----------------------|-------------------------------------------------------------------------------------------------|
| resolution            | How many cells the images will be divided into. Higher resolution will capture more high frequency details. |
| proximity importance  | How much the algorithm changes the original image to make it look like the target image. Increase this if you want a more subtle transformation. |
| algorithm             | The algorithm used to calculate the assignment of each pixel. Optimal will find the mathematically optimal solution, but is extremely slow for high resolutions. |

# Installations

Install the latest version in [releases](https://github.com/Spu7Nix/obamify/releases). Unzip and run the .exe file inside!
**Note for macOS users:**
Run 'xattr -C <path/to/app.app>' in your terminal to remove the damaged app warning. 
### Building from source

1. Install [Rust](https://www.rust-lang.org/tools/install)
2. Run `cargo run --release` in the project folder

#### Running the web version locally
1. Install [Rust](https://www.rust-lang.org/tools/install)
2. Install the required target with `rustup target add wasm32-unknown-unknown`
3. Install Trunk with `cargo install --locked trunk`
4. Run `trunk serve --release --open`

# Contributing

Please open an issue or a pull request if you have any suggestions or find any bugs :)

# How it works

black magic
