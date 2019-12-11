import numpy as np
from kivy.app import App
from kivy.uix.floatlayout import FloatLayout
from kivy.clock import Clock
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.behaviors import ButtonBehavior
from kivy.uix.image import Image
from kivy.properties import NumericProperty, BooleanProperty, StringProperty
from kivy.uix.screenmanager import ScreenManager, Screen
from SantoGame import SantoriniEnv
from kivy.core.window import Window
Window.size = (600, 800)
from SantoAI import QLearningAgent
# Window.clearcolor = (1, 1, 1, 1)
import time

class SantoGUI(FloatLayout):

    def __init__(self, env):
        super(SantoGUI, self).__init__()
        self.env = env
        if (self.env.size == 9) | (self.env.size == 6):
            self.x_dict = {0: 0, 1: 1/3, 2: 2/3, 3: 0, 4: 1/3, 5: 2/3, 6: 0, 7: 1/3, 8: 2/3}
            self.y_dict = {0: 0, 1: 0, 2: 0, 3: 0.25, 4: 0.25, 5: 0.25, 6: 0.5, 7: 0.5, 8: 0.5}
        else:
            self.x_dict = {0: 0, 1: 1/2, 2: 0, 3: 1/2}
            self.y_dict = {0: 0, 1: 0, 2: 1/3, 3: 1/3}
        for button_nr in range(self.env.size):
            x_pos = self.x_dict[button_nr]
            y_pos = self.y_dict[button_nr]
            self.add_widget(SantoButton(env, id=str(button_nr), pos_hint={'x': x_pos, 'y': y_pos}))
        self.add_widget(Label(id='CurrentPlayerLabel', pos_hint={'x': 0, 'y': 0.75}, size_hint=(0.5, 0.25),
                               font_size=30, text='Current Player: ' + str(self.env.current_player)))
        self.add_widget(Label(id='CurrentPhaseLabel', pos_hint={'x': 0.5, 'y': 0.75}, size_hint=(0.5, 0.25),
                              font_size=30, text='Phase: ' + str(self.env.phase)))

    def start(self):
        self.run = Clock.schedule_interval(self.update, 2.0 / 60.0)

    def update(self, dt):
        """
        Paint job:
        For every square:
        - If a viable action can be done here -> green
        - If player 1 here -> Red
        - If player 2 here -> Blue
        :param dt:
        :return:
        """
        self.paint_building_level()
        self.update_labels()
        if self.env.end_condition()[0]:
            time.sleep(1)
            self.run.cancel()
            if self.env.end_condition()[1] == 1:
                self.parent.manager.winner = 'Victory\nClick to play again!'
            elif self.env.end_condition()[1] == -1:
                self.parent.manager.winner = 'Loss\nClick to play again!'
            self.parent.manager.current = 'END'

    def update_labels(self):
        for child_widget in self.children:
            if 'Label' in child_widget.id:
                if child_widget.id == 'CurrentPlayerLabel':
                    child_widget.text = 'Current Player: ' + str(self.env.current_player)
                elif child_widget.id == 'CurrentPhaseLabel':
                    child_widget.text = 'Phase: ' + str(self.env.phase)


    def paint_building_level(self):
        viable_actions = self.env.get_viable_actions(self.env.state)
        viable_coords = [self.env.target_position_given_action(a, self.env.state) for a in viable_actions]
        viable_buttons = [str(self.env.inv_coords[c]) for c in viable_coords]
        p1_button, p2_button = self.env.player_positions[1], self.env.player_positions[2]
        for child_widget in self.children:
            if 'Label' not in child_widget.id:  # If not a label, it is a button
                build_state_str = 'Build' + child_widget.id
                image_str = 'level_' + str(self.env.state[self.env.state_index_dict[build_state_str]]) + '_'
                if child_widget.id == str(p1_button):
                    image_str += 'red_'
                elif child_widget.id == str(p2_button):
                    image_str += 'gray_'
                else:
                    image_str += 'empty_'
                if child_widget.id in ['1', '3', '5', '7', '9']:
                    image_str += 'dark_'
                else:
                    image_str += 'light_'
                if child_widget.id in viable_buttons:
                    image_str += 'eligible'
                else:
                    image_str += 'neutral'
                image_str += '.PNG'
                child_widget.source = image_str


class SantoButton(ButtonBehavior, Image):

    def __init__(self, env, **kwargs):
        self.env = env
        super(SantoButton, self).__init__(**kwargs)
        self.font_size = 30
        self.color = [1, 1, 1, 1]
        if self.env.size == 9:
            self.size_hint = (1/3, 0.25)
        elif self.env.size == 4:
            self.size_hint = (1/2, 3/8)
        self.source = 'Level0.png'

    def on_touch_down(self, touch):
        if self.collide_point(*touch.pos):
            target_pos = int(self.id)
            if self.env.current_player == 1:
                action = self.env.action_given_target_position(target_pos)
                if action is None:
                    pass
                else:
                    _, _, done = self.env.step_play(action)
            else:
                action = None
                _, _, done = self.env.step_play(action)


class CustomScreenManager(ScreenManager):
    winner = StringProperty(True)


class TitleScreen(Screen):
    def __init__(self, name):
        super(TitleScreen, self).__init__()
        self.name = name

    def on_enter(self, *args):
        self.add_widget(Button(text='Click To Play', font_size=50))

    def on_leave(self, *args):
        self.clear_widgets()

    def on_touch_down(self, touch):
        self.manager.current = 'GAME'


class GameScreen(Screen):
    def __init__(self, gui, name):
        super(GameScreen, self).__init__()
        self.name = name
        self.gui = gui

    def on_enter(self, *args):
        self.gui.env.reset()
        self.gui.start()
        self.add_widget(gui)

    def on_leave(self, *args):
        self.clear_widgets()


class EndScreen(Screen):
    def __init__(self, name):
        super(EndScreen, self).__init__()
        self.name = name

    def on_enter(self, *args):
        text = self.parent.winner
        self.add_widget(Button(text=text, font_size=50))

    def on_leave(self, *args):
        self.clear_widgets()

    def on_touch_down(self, touch):
        self.manager.current = 'GAME'

env_instance = SantoriniEnv()
agent = QLearningAgent(lr=None, gamma=None, epsilon=0,
                       get_legal_actions=env_instance.get_viable_actions, filepath='QDerpo500krb16.p')
env_instance.opponent_agent = agent
gui = SantoGUI(env_instance)

sm = CustomScreenManager()
sm.add_widget(TitleScreen(name='TITLE'))
sm.add_widget(GameScreen(gui, name='GAME'))
sm.add_widget(EndScreen(name='END'))
sm.current = 'TITLE'


class SantoGUIApp(App):

    def build(self):
        return sm


app = SantoGUIApp()
app.run()