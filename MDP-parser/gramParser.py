# Generated from gram.g4 by ANTLR 4.7.2
# encoding: utf-8
from antlr4 import *
from io import StringIO
from typing.io import TextIO
import sys

def serializedATN():
    with StringIO() as buf:
        buf.write("\3\u608b\ua72a\u8133\ub9ed\u417c\u3be7\u7786\u5964\3\17")
        buf.write("o\4\2\t\2\4\3\t\3\4\4\t\4\4\5\t\5\4\6\t\6\4\7\t\7\4\b")
        buf.write("\t\b\4\t\t\t\4\n\t\n\3\2\3\2\3\2\3\2\3\2\3\3\3\3\5\3\34")
        buf.write("\n\3\3\4\3\4\3\4\3\4\7\4\"\n\4\f\4\16\4%\13\4\3\4\3\4")
        buf.write("\3\5\3\5\3\5\3\5\3\5\3\5\7\5/\n\5\f\5\16\5\62\13\5\3\5")
        buf.write("\3\5\3\6\3\6\3\6\3\6\7\6:\n\6\f\6\16\6=\13\6\3\6\3\6\3")
        buf.write("\7\3\7\7\7C\n\7\f\7\16\7F\13\7\3\b\3\b\5\bJ\n\b\3\t\3")
        buf.write("\t\3\t\3\t\3\t\3\t\3\t\3\t\3\t\3\t\3\t\3\t\7\tX\n\t\f")
        buf.write("\t\16\t[\13\t\3\t\3\t\3\n\3\n\3\n\3\n\3\n\3\n\3\n\3\n")
        buf.write("\3\n\7\nh\n\n\f\n\16\nk\13\n\3\n\3\n\3\n\2\2\13\2\4\6")
        buf.write("\b\n\f\16\20\22\2\2\2m\2\24\3\2\2\2\4\33\3\2\2\2\6\35")
        buf.write("\3\2\2\2\b(\3\2\2\2\n\65\3\2\2\2\f@\3\2\2\2\16I\3\2\2")
        buf.write("\2\20K\3\2\2\2\22^\3\2\2\2\24\25\5\4\3\2\25\26\5\n\6\2")
        buf.write("\26\27\5\f\7\2\27\30\7\2\2\3\30\3\3\2\2\2\31\34\5\6\4")
        buf.write("\2\32\34\5\b\5\2\33\31\3\2\2\2\33\32\3\2\2\2\34\5\3\2")
        buf.write("\2\2\35\36\7\3\2\2\36#\7\16\2\2\37 \7\t\2\2 \"\7\16\2")
        buf.write("\2!\37\3\2\2\2\"%\3\2\2\2#!\3\2\2\2#$\3\2\2\2$&\3\2\2")
        buf.write("\2%#\3\2\2\2&\'\7\b\2\2\'\7\3\2\2\2()\7\3\2\2)\60\7\16")
        buf.write("\2\2*+\7\t\2\2+,\7\16\2\2,-\7\6\2\2-/\7\r\2\2.*\3\2\2")
        buf.write("\2/\62\3\2\2\2\60.\3\2\2\2\60\61\3\2\2\2\61\63\3\2\2\2")
        buf.write("\62\60\3\2\2\2\63\64\7\b\2\2\64\t\3\2\2\2\65\66\7\4\2")
        buf.write("\2\66;\7\16\2\2\678\7\t\2\28:\7\16\2\29\67\3\2\2\2:=\3")
        buf.write("\2\2\2;9\3\2\2\2;<\3\2\2\2<>\3\2\2\2=;\3\2\2\2>?\7\b\2")
        buf.write("\2?\13\3\2\2\2@D\5\16\b\2AC\5\16\b\2BA\3\2\2\2CF\3\2\2")
        buf.write("\2DB\3\2\2\2DE\3\2\2\2E\r\3\2\2\2FD\3\2\2\2GJ\5\20\t\2")
        buf.write("HJ\5\22\n\2IG\3\2\2\2IH\3\2\2\2J\17\3\2\2\2KL\7\16\2\2")
        buf.write("LM\7\13\2\2MN\7\16\2\2NO\7\f\2\2OP\7\7\2\2PQ\7\r\2\2Q")
        buf.write("R\7\6\2\2RY\7\16\2\2ST\7\n\2\2TU\7\r\2\2UV\7\6\2\2VX\7")
        buf.write("\16\2\2WS\3\2\2\2X[\3\2\2\2YW\3\2\2\2YZ\3\2\2\2Z\\\3\2")
        buf.write("\2\2[Y\3\2\2\2\\]\7\b\2\2]\21\3\2\2\2^_\7\16\2\2_`\7\7")
        buf.write("\2\2`a\7\r\2\2ab\7\6\2\2bi\7\16\2\2cd\7\n\2\2de\7\r\2")
        buf.write("\2ef\7\6\2\2fh\7\16\2\2gc\3\2\2\2hk\3\2\2\2ig\3\2\2\2")
        buf.write("ij\3\2\2\2jl\3\2\2\2ki\3\2\2\2lm\7\b\2\2m\23\3\2\2\2\n")
        buf.write("\33#\60;DIYi")
        return buf.getvalue()


class gramParser ( Parser ):

    grammarFileName = "gram.g4"

    atn = ATNDeserializer().deserialize(serializedATN())

    decisionsToDFA = [ DFA(ds, i) for i, ds in enumerate(atn.decisionToState) ]

    sharedContextCache = PredictionContextCache()

    literalNames = [ "<INVALID>", "'States'", "'Actions'", "'transition'", 
                     "':'", "'->'", "';'", "','", "'+'", "'['", "']'" ]

    symbolicNames = [ "<INVALID>", "STATES", "ACTIONS", "TRANSITION", "DPOINT", 
                      "FLECHE", "SEMI", "VIRG", "PLUS", "LCROCH", "RCROCH", 
                      "INT", "ID", "WS" ]

    RULE_program = 0
    RULE_defstates = 1
    RULE_statenoreward = 2
    RULE_statereward = 3
    RULE_defactions = 4
    RULE_transitions = 5
    RULE_trans = 6
    RULE_transact = 7
    RULE_transnoact = 8

    ruleNames =  [ "program", "defstates", "statenoreward", "statereward", 
                   "defactions", "transitions", "trans", "transact", "transnoact" ]

    EOF = Token.EOF
    STATES=1
    ACTIONS=2
    TRANSITION=3
    DPOINT=4
    FLECHE=5
    SEMI=6
    VIRG=7
    PLUS=8
    LCROCH=9
    RCROCH=10
    INT=11
    ID=12
    WS=13

    def __init__(self, input:TokenStream, output:TextIO = sys.stdout):
        super().__init__(input, output)
        self.checkVersion("4.7.2")
        self._interp = ParserATNSimulator(self, self.atn, self.decisionsToDFA, self.sharedContextCache)
        self._predicates = None



    class ProgramContext(ParserRuleContext):

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def defstates(self):
            return self.getTypedRuleContext(gramParser.DefstatesContext,0)


        def defactions(self):
            return self.getTypedRuleContext(gramParser.DefactionsContext,0)


        def transitions(self):
            return self.getTypedRuleContext(gramParser.TransitionsContext,0)


        def EOF(self):
            return self.getToken(gramParser.EOF, 0)

        def getRuleIndex(self):
            return gramParser.RULE_program

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterProgram" ):
                listener.enterProgram(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitProgram" ):
                listener.exitProgram(self)




    def program(self):

        localctx = gramParser.ProgramContext(self, self._ctx, self.state)
        self.enterRule(localctx, 0, self.RULE_program)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 18
            self.defstates()
            self.state = 19
            self.defactions()
            self.state = 20
            self.transitions()
            self.state = 21
            self.match(gramParser.EOF)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class DefstatesContext(ParserRuleContext):

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def statenoreward(self):
            return self.getTypedRuleContext(gramParser.StatenorewardContext,0)


        def statereward(self):
            return self.getTypedRuleContext(gramParser.StaterewardContext,0)


        def getRuleIndex(self):
            return gramParser.RULE_defstates

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterDefstates" ):
                listener.enterDefstates(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitDefstates" ):
                listener.exitDefstates(self)




    def defstates(self):

        localctx = gramParser.DefstatesContext(self, self._ctx, self.state)
        self.enterRule(localctx, 2, self.RULE_defstates)
        try:
            self.state = 25
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input,0,self._ctx)
            if la_ == 1:
                self.enterOuterAlt(localctx, 1)
                self.state = 23
                self.statenoreward()
                pass

            elif la_ == 2:
                self.enterOuterAlt(localctx, 2)
                self.state = 24
                self.statereward()
                pass


        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class StatenorewardContext(ParserRuleContext):

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def STATES(self):
            return self.getToken(gramParser.STATES, 0)

        def ID(self, i:int=None):
            if i is None:
                return self.getTokens(gramParser.ID)
            else:
                return self.getToken(gramParser.ID, i)

        def SEMI(self):
            return self.getToken(gramParser.SEMI, 0)

        def VIRG(self, i:int=None):
            if i is None:
                return self.getTokens(gramParser.VIRG)
            else:
                return self.getToken(gramParser.VIRG, i)

        def getRuleIndex(self):
            return gramParser.RULE_statenoreward

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterStatenoreward" ):
                listener.enterStatenoreward(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitStatenoreward" ):
                listener.exitStatenoreward(self)




    def statenoreward(self):

        localctx = gramParser.StatenorewardContext(self, self._ctx, self.state)
        self.enterRule(localctx, 4, self.RULE_statenoreward)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 27
            self.match(gramParser.STATES)
            self.state = 28
            self.match(gramParser.ID)
            self.state = 33
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            while _la==gramParser.VIRG:
                self.state = 29
                self.match(gramParser.VIRG)
                self.state = 30
                self.match(gramParser.ID)
                self.state = 35
                self._errHandler.sync(self)
                _la = self._input.LA(1)

            self.state = 36
            self.match(gramParser.SEMI)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class StaterewardContext(ParserRuleContext):

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def STATES(self):
            return self.getToken(gramParser.STATES, 0)

        def ID(self, i:int=None):
            if i is None:
                return self.getTokens(gramParser.ID)
            else:
                return self.getToken(gramParser.ID, i)

        def SEMI(self):
            return self.getToken(gramParser.SEMI, 0)

        def VIRG(self, i:int=None):
            if i is None:
                return self.getTokens(gramParser.VIRG)
            else:
                return self.getToken(gramParser.VIRG, i)

        def DPOINT(self, i:int=None):
            if i is None:
                return self.getTokens(gramParser.DPOINT)
            else:
                return self.getToken(gramParser.DPOINT, i)

        def INT(self, i:int=None):
            if i is None:
                return self.getTokens(gramParser.INT)
            else:
                return self.getToken(gramParser.INT, i)

        def getRuleIndex(self):
            return gramParser.RULE_statereward

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterStatereward" ):
                listener.enterStatereward(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitStatereward" ):
                listener.exitStatereward(self)




    def statereward(self):

        localctx = gramParser.StaterewardContext(self, self._ctx, self.state)
        self.enterRule(localctx, 6, self.RULE_statereward)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 38
            self.match(gramParser.STATES)
            self.state = 39
            self.match(gramParser.ID)
            self.state = 46
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            while _la==gramParser.VIRG:
                self.state = 40
                self.match(gramParser.VIRG)
                self.state = 41
                self.match(gramParser.ID)
                self.state = 42
                self.match(gramParser.DPOINT)
                self.state = 43
                self.match(gramParser.INT)
                self.state = 48
                self._errHandler.sync(self)
                _la = self._input.LA(1)

            self.state = 49
            self.match(gramParser.SEMI)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class DefactionsContext(ParserRuleContext):

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def ACTIONS(self):
            return self.getToken(gramParser.ACTIONS, 0)

        def ID(self, i:int=None):
            if i is None:
                return self.getTokens(gramParser.ID)
            else:
                return self.getToken(gramParser.ID, i)

        def SEMI(self):
            return self.getToken(gramParser.SEMI, 0)

        def VIRG(self, i:int=None):
            if i is None:
                return self.getTokens(gramParser.VIRG)
            else:
                return self.getToken(gramParser.VIRG, i)

        def getRuleIndex(self):
            return gramParser.RULE_defactions

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterDefactions" ):
                listener.enterDefactions(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitDefactions" ):
                listener.exitDefactions(self)




    def defactions(self):

        localctx = gramParser.DefactionsContext(self, self._ctx, self.state)
        self.enterRule(localctx, 8, self.RULE_defactions)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 51
            self.match(gramParser.ACTIONS)
            self.state = 52
            self.match(gramParser.ID)
            self.state = 57
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            while _la==gramParser.VIRG:
                self.state = 53
                self.match(gramParser.VIRG)
                self.state = 54
                self.match(gramParser.ID)
                self.state = 59
                self._errHandler.sync(self)
                _la = self._input.LA(1)

            self.state = 60
            self.match(gramParser.SEMI)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class TransitionsContext(ParserRuleContext):

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def trans(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(gramParser.TransContext)
            else:
                return self.getTypedRuleContext(gramParser.TransContext,i)


        def getRuleIndex(self):
            return gramParser.RULE_transitions

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterTransitions" ):
                listener.enterTransitions(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitTransitions" ):
                listener.exitTransitions(self)




    def transitions(self):

        localctx = gramParser.TransitionsContext(self, self._ctx, self.state)
        self.enterRule(localctx, 10, self.RULE_transitions)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 62
            self.trans()
            self.state = 66
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            while _la==gramParser.ID:
                self.state = 63
                self.trans()
                self.state = 68
                self._errHandler.sync(self)
                _la = self._input.LA(1)

        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class TransContext(ParserRuleContext):

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def transact(self):
            return self.getTypedRuleContext(gramParser.TransactContext,0)


        def transnoact(self):
            return self.getTypedRuleContext(gramParser.TransnoactContext,0)


        def getRuleIndex(self):
            return gramParser.RULE_trans

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterTrans" ):
                listener.enterTrans(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitTrans" ):
                listener.exitTrans(self)




    def trans(self):

        localctx = gramParser.TransContext(self, self._ctx, self.state)
        self.enterRule(localctx, 12, self.RULE_trans)
        try:
            self.state = 71
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input,5,self._ctx)
            if la_ == 1:
                self.enterOuterAlt(localctx, 1)
                self.state = 69
                self.transact()
                pass

            elif la_ == 2:
                self.enterOuterAlt(localctx, 2)
                self.state = 70
                self.transnoact()
                pass


        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class TransactContext(ParserRuleContext):

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def ID(self, i:int=None):
            if i is None:
                return self.getTokens(gramParser.ID)
            else:
                return self.getToken(gramParser.ID, i)

        def LCROCH(self):
            return self.getToken(gramParser.LCROCH, 0)

        def RCROCH(self):
            return self.getToken(gramParser.RCROCH, 0)

        def FLECHE(self):
            return self.getToken(gramParser.FLECHE, 0)

        def INT(self, i:int=None):
            if i is None:
                return self.getTokens(gramParser.INT)
            else:
                return self.getToken(gramParser.INT, i)

        def DPOINT(self, i:int=None):
            if i is None:
                return self.getTokens(gramParser.DPOINT)
            else:
                return self.getToken(gramParser.DPOINT, i)

        def SEMI(self):
            return self.getToken(gramParser.SEMI, 0)

        def PLUS(self, i:int=None):
            if i is None:
                return self.getTokens(gramParser.PLUS)
            else:
                return self.getToken(gramParser.PLUS, i)

        def getRuleIndex(self):
            return gramParser.RULE_transact

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterTransact" ):
                listener.enterTransact(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitTransact" ):
                listener.exitTransact(self)




    def transact(self):

        localctx = gramParser.TransactContext(self, self._ctx, self.state)
        self.enterRule(localctx, 14, self.RULE_transact)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 73
            self.match(gramParser.ID)
            self.state = 74
            self.match(gramParser.LCROCH)
            self.state = 75
            self.match(gramParser.ID)
            self.state = 76
            self.match(gramParser.RCROCH)
            self.state = 77
            self.match(gramParser.FLECHE)
            self.state = 78
            self.match(gramParser.INT)
            self.state = 79
            self.match(gramParser.DPOINT)
            self.state = 80
            self.match(gramParser.ID)
            self.state = 87
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            while _la==gramParser.PLUS:
                self.state = 81
                self.match(gramParser.PLUS)
                self.state = 82
                self.match(gramParser.INT)
                self.state = 83
                self.match(gramParser.DPOINT)
                self.state = 84
                self.match(gramParser.ID)
                self.state = 89
                self._errHandler.sync(self)
                _la = self._input.LA(1)

            self.state = 90
            self.match(gramParser.SEMI)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class TransnoactContext(ParserRuleContext):

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def ID(self, i:int=None):
            if i is None:
                return self.getTokens(gramParser.ID)
            else:
                return self.getToken(gramParser.ID, i)

        def FLECHE(self):
            return self.getToken(gramParser.FLECHE, 0)

        def INT(self, i:int=None):
            if i is None:
                return self.getTokens(gramParser.INT)
            else:
                return self.getToken(gramParser.INT, i)

        def DPOINT(self, i:int=None):
            if i is None:
                return self.getTokens(gramParser.DPOINT)
            else:
                return self.getToken(gramParser.DPOINT, i)

        def SEMI(self):
            return self.getToken(gramParser.SEMI, 0)

        def PLUS(self, i:int=None):
            if i is None:
                return self.getTokens(gramParser.PLUS)
            else:
                return self.getToken(gramParser.PLUS, i)

        def getRuleIndex(self):
            return gramParser.RULE_transnoact

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterTransnoact" ):
                listener.enterTransnoact(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitTransnoact" ):
                listener.exitTransnoact(self)




    def transnoact(self):

        localctx = gramParser.TransnoactContext(self, self._ctx, self.state)
        self.enterRule(localctx, 16, self.RULE_transnoact)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 92
            self.match(gramParser.ID)
            self.state = 93
            self.match(gramParser.FLECHE)
            self.state = 94
            self.match(gramParser.INT)
            self.state = 95
            self.match(gramParser.DPOINT)
            self.state = 96
            self.match(gramParser.ID)
            self.state = 103
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            while _la==gramParser.PLUS:
                self.state = 97
                self.match(gramParser.PLUS)
                self.state = 98
                self.match(gramParser.INT)
                self.state = 99
                self.match(gramParser.DPOINT)
                self.state = 100
                self.match(gramParser.ID)
                self.state = 105
                self._errHandler.sync(self)
                _la = self._input.LA(1)

            self.state = 106
            self.match(gramParser.SEMI)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx





