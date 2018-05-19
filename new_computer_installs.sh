xcode-select --install

/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
brew doctor # after macOS or xcode update
brew tap caskroom/cask

brew install python  
brew install python@2
brew install git
brew install wget
brew install tmux
brew install docker
# brew install docker-machine # tool that can install and manage Docker Engine on virtual hosts
# brew install docker-compose # tool for defining and running multi-container docker applications
brew install awscli
brew install aws-shell  # accepts the same commands as the AWS CLI, except you don't need to provide the aws prefix.
brew install node # for awscreds
npm install -g https://mvnrepo.nordstrom.net/nexus/content/groups/NPMgroup/awscreds/-/awscreds-1.1.1.tgz
brew cask install docker
brew cask install iterm2
brew install bash-completion
brew cask install sqlworkbenchj
brew cask install dbeaver-community
brew cask install atom
# apm install python-autopep8   # style guide check
# apm install platformio-ide-terminal   # run scripts in atom, within venv?
# apm install python-debugger
# apm install data-atom         # Query and manage databases from within Atom to run SQL files.
#                               # To connect to a database, type Alt-Shift-R. Type your script in Atom, then either click Execute or F5.
# apm install merge-conflicts   # to resolve merge conflicts in git
# apm install remote-sync       # https://atom.io/packages/remote-sync
brew cask install google-chrome
brew cask install flux
brew cask install slack
sudo periodic daily weekly monthly  # runs periodic checks on command

# brew install tree
# brew install htop
# brew install macvim
# brew install tor
# brew cask install mactex
# brew cask install adobe-acrobat-reader
# brew cask install blue-jeans # use in browser

# brew cask install java
# Note: If using `r` and `rJava` the `brew cask` version of java will not work.
# You want to get rid of that and install the JDK version of java 8 from:
# http://www.oracle.com/technetwork/java/javase/downloads/jdk8-downloads-2133151.html
# brew install r
# brew install r-app
# brew cask install rstudio
# brew cask install xquartz     # better graphics in r
# apm install language-r        # atom plugin for .R scripts
